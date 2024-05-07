﻿// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Http;
using Microsoft.SemanticKernel.Text;

namespace Microsoft.SemanticKernel.Connectors.HuggingFace.Core;

/// <summary>
/// This class is responsible for making HTTP requests to the HuggingFace Inference API - Chat Completion Message API
/// <see href="https://huggingface.co/docs/text-generation-inference/main/en/messages_api" />
/// </summary>
internal sealed class HuggingFaceMessageApiClient
{
    private readonly HuggingFaceClient _clientCore;

    private static readonly string s_namespace = typeof(HuggingFaceMessageApiClient).Namespace!;

    /// <summary>
    /// Instance of <see cref="Meter"/> for metrics.
    /// </summary>
    private static readonly Meter s_meter = new(s_namespace);

    /// <summary>
    /// Instance of <see cref="Counter{T}"/> to keep track of the number of prompt tokens used.
    /// </summary>
    private static readonly Counter<int> s_promptTokensCounter =
        s_meter.CreateCounter<int>(
            name: $"{s_namespace}.tokens.prompt",
            unit: "{token}",
            description: "Number of prompt tokens used");

    /// <summary>
    /// Instance of <see cref="Counter{T}"/> to keep track of the number of completion tokens used.
    /// </summary>
    private static readonly Counter<int> s_completionTokensCounter =
        s_meter.CreateCounter<int>(
            name: $"{s_namespace}.tokens.completion",
            unit: "{token}",
            description: "Number of completion tokens used");

    /// <summary>
    /// Instance of <see cref="Counter{T}"/> to keep track of the total number of tokens used.
    /// </summary>
    private static readonly Counter<int> s_totalTokensCounter =
        s_meter.CreateCounter<int>(
            name: $"{s_namespace}.tokens.total",
            unit: "{token}",
            description: "Number of total tokens used");

    internal HuggingFaceMessageApiClient(
        string modelId,
        HttpClient httpClient,
        Uri? endpoint = null,
        string? apiKey = null,
        ILogger? logger = null)
    {
        this._clientCore = new(
            modelId,
            httpClient,
            endpoint,
            apiKey,
            logger);
    }

    internal async IAsyncEnumerable<StreamingChatMessageContent> StreamCompleteChatMessageAsync(
      ChatHistory chatHistory,
      PromptExecutionSettings? executionSettings,
      [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        string modelId = executionSettings?.ModelId ?? this._clientCore.ModelId;
        var endpoint = this.GetChatGenerationEndpoint();
        var request = this.CreateChatRequest(chatHistory, executionSettings);
        request.Stream = true;

        using var httpRequestMessage = this._clientCore.CreatePost(request, endpoint, this._clientCore.ApiKey);

        using var response = await this._clientCore.SendRequestAndGetResponseImmediatelyAfterHeadersReadAsync(httpRequestMessage, cancellationToken)
            .ConfigureAwait(false);

        using var responseStream = await response.Content.ReadAsStreamAndTranslateExceptionAsync()
            .ConfigureAwait(false);

        await foreach (var streamingChatContent in this.ProcessChatResponseStreamAsync(responseStream, modelId, cancellationToken).ConfigureAwait(false))
        {
            yield return streamingChatContent;
        }
    }

    internal async Task<IReadOnlyList<ChatMessageContent>> CompleteChatMessageAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings,
        CancellationToken cancellationToken)
    {
        string modelId = executionSettings?.ModelId ?? this._clientCore.ModelId;
        var endpoint = this.GetChatGenerationEndpoint();
        var request = this.CreateChatRequest(chatHistory, executionSettings);
        using var httpRequestMessage = this._clientCore.CreatePost(request, endpoint, this._clientCore.ApiKey);

        string body = await this._clientCore.SendRequestAndGetStringBodyAsync(httpRequestMessage, cancellationToken)
            .ConfigureAwait(false);

        var response = HuggingFaceClient.DeserializeResponse<ChatCompletionResponse>(body);
        var chatContents = GetChatMessageContentsFromResponse(response, modelId);

        this.LogChatCompletionUsage(executionSettings, response);

        return chatContents;
    }

    private void LogChatCompletionUsage(PromptExecutionSettings? executionSettings, ChatCompletionResponse chatCompletionResponse)
    {
        if (this._clientCore.Logger.IsEnabled(LogLevel.Debug))
        {
            this._clientCore.Logger.Log(
            LogLevel.Debug,
            "HuggingFace chat completion usage - ModelId: {ModelId}, Prompt tokens: {PromptTokens}, Completion tokens: {CompletionTokens}, Total tokens: {TotalTokens}",
            chatCompletionResponse.Model,
            chatCompletionResponse.Usage!.PromptTokens,
            chatCompletionResponse.Usage!.CompletionTokens,
            chatCompletionResponse.Usage!.TotalTokens);
        }

        s_promptTokensCounter.Add(chatCompletionResponse.Usage!.PromptTokens);
        s_completionTokensCounter.Add(chatCompletionResponse.Usage!.CompletionTokens);
        s_totalTokensCounter.Add(chatCompletionResponse.Usage!.TotalTokens);
    }

    private static List<ChatMessageContent> GetChatMessageContentsFromResponse(ChatCompletionResponse response, string modelId)
    {
        var chatMessageContents = new List<ChatMessageContent>();

        foreach (var choice in response.Choices!)
        {
            var metadata = new HuggingFaceChatCompletionMetadata
            {
                Id = response.Id,
                Model = response.Model,
                @Object = response.Object,
                SystemFingerPrint = response.SystemFingerprint,
                Created = response.Created,
                FinishReason = choice.FinishReason,
                LogProbs = choice.LogProbs,
                UsageCompletionTokens = response.Usage?.CompletionTokens,
                UsagePromptTokens = response.Usage?.PromptTokens,
                UsageTotalTokens = response.Usage?.TotalTokens,
            };

            chatMessageContents.Add(new ChatMessageContent(
                role: new AuthorRole(choice.Message?.Role ?? AuthorRole.Assistant.ToString()),
                content: choice.Message?.Content,
                modelId: response.Model,
                innerContent: response,
                encoding: Encoding.UTF8,
                metadata: metadata));
        }

        return chatMessageContents;
    }

    private static StreamingChatMessageContent GetStreamingChatMessageContentFromStreamResponse(ChatCompletionStreamResponse response, string modelId)
    {
        var choice = response.Choices.FirstOrDefault();
        if (choice is not null)
        {
            var metadata = new HuggingFaceChatCompletionMetadata
            {
                Id = response.Id,
                Model = response.Model,
                @Object = response.Object,
                SystemFingerPrint = response.SystemFingerprint,
                Created = response.Created,
                FinishReason = choice.FinishReason,
                LogProbs = choice.LogProbs,
            };

            var streamChat = new StreamingChatMessageContent(
                choice.Delta?.Role is not null ? new AuthorRole(choice.Delta.Role) : null,
                choice.Delta?.Content,
                response,
                choice.Index,
                modelId,
                Encoding.UTF8,
                metadata);

            return streamChat;
        }

        throw new KernelException("Unexpected response from model")
        {
            Data = { { "ResponseData", response } },
        };
    }

    private async IAsyncEnumerable<StreamingChatMessageContent> ProcessChatResponseStreamAsync(Stream stream, string modelId, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var content in this.ParseChatResponseStreamAsync(stream, cancellationToken).ConfigureAwait(false))
        {
            yield return GetStreamingChatMessageContentFromStreamResponse(content, modelId);
        }
    }

    private ChatCompletionRequest CreateChatRequest(
        ChatHistory chatHistory,
        PromptExecutionSettings? promptExecutionSettings)
    {
        var huggingFaceExecutionSettings = HuggingFacePromptExecutionSettings.FromExecutionSettings(promptExecutionSettings);
        huggingFaceExecutionSettings.ModelId ??= this._clientCore.ModelId;

        HuggingFaceClient.ValidateMaxTokens(huggingFaceExecutionSettings.MaxTokens);
        var request = ChatCompletionRequest.FromChatHistoryAndExecutionSettings(chatHistory, huggingFaceExecutionSettings);
        return request;
    }

    private IAsyncEnumerable<ChatCompletionStreamResponse> ParseChatResponseStreamAsync(Stream responseStream, CancellationToken cancellationToken)
        => SseJsonParser.ParseAsync<ChatCompletionStreamResponse>(responseStream, cancellationToken);

    private Uri GetChatGenerationEndpoint()
        => new($"{this._clientCore.Endpoint}{this._clientCore.Separator}v1/chat/completions");
}
