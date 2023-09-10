import json
import os

import django

from chat.agents.agent_factory import AgentFactory
from chat.agents.callbacks import AsyncStreamingCallbackHandler
from chat.messages.chat_message_repository import ChatMessageRepository
from huggingface_hub import AsyncInferenceClient

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from channels.generic.websocket import AsyncWebsocketConsumer
from langchain.agents import AgentExecutor

from chat.models import MessageSender


class ChatConsumer(AsyncWebsocketConsumer):
    # The LLM agent for this chat application
    agent: AgentExecutor
    inference_client: AsyncInferenceClient

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_factory = AgentFactory()
        self.chat_message_repository = ChatMessageRepository()

    async def connect(self):
        # Get the chat_id from the client
        chat_id = self.scope['url_route']['kwargs'].get('chat_id')

        # Create the agent when the websocket connection with the client is established
        self.agent = await self.agent_factory.create_agent(
            tool_names=["llm-math"],
            chat_id=chat_id,
            streaming=True,
            callback_handlers=[AsyncStreamingCallbackHandler(self)],
        )
        self.inference_client = await self.agent_factory.create_client()

        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        chat_id = text_data_json['chat_id']

        # Forward the message to LangChain
        response = await self.message_agent(message, chat_id)

        # Send the response from the OpenAI Chat API to the frontend client
        await self.send(text_data=json.dumps({'message': response, 'type': 'answer'}))

    async def message_agent(self, message: str, chat_id: str):
        # Save the user message to the database
        await self.chat_message_repository.save_message(message=message, sender=MessageSender.USER.value, chat_id=chat_id)

        # Call the agent
        response = await self.agent.arun(message)

        # Save the AI message to the database
        await self.chat_message_repository.save_message(message=response, sender=MessageSender.AI.value, chat_id=chat_id)

        return response

    async def message_client(self, message: str, chat_id: str):
        # Save the user message to the database
        await self.chat_message_repository.save_message(message=message, sender=MessageSender.USER.value,
                                                        chat_id=chat_id)

        prompt = f"Instructions: Make your answer less than 500 words. Question: {message}\n Answer: "

        generated_text = await self.inference_client.text_generation(prompt=prompt, max_new_tokens=420)
        # generated_text += token  # Concatenate each token
        print(generated_text, end="")

        # Save the AI message to the database
        await self.chat_message_repository.save_message(message=generated_text, sender=MessageSender.AI.value,
                                                        chat_id=chat_id)

        return generated_text

    def my_callback(self, message):
        print("Callback received:", message)
