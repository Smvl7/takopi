import pytest
from takopi.config import ConfigError
from takopi.settings import TelegramTopicsSettings
from takopi.telegram.api_schemas import User, Chat, ChatMember
from takopi.telegram.topics import _validate_topics_setup_for

class MockBot:
    def __init__(self, user, chats, members):
        self._user = user
        self._chats = chats
        self._members = members

    async def get_me(self):
        return self._user
    
    async def get_chat(self, chat_id):
        return self._chats.get(chat_id)
    
    async def get_chat_member(self, chat_id, bot_id):
        return self._members.get((chat_id, bot_id))

@pytest.mark.anyio
async def test_validate_topics_private_chat_success():
    bot = MockBot(
        user=User(id=1, has_topics_enabled=True),
        chats={100: Chat(id=100, type="private")},
        members={}
    )
    settings = TelegramTopicsSettings(enabled=True, scope="main")
    # should not raise
    await _validate_topics_setup_for(
        bot=bot,
        topics=settings,
        chat_id=100,
        project_chat_ids=[]
    )

@pytest.mark.anyio
async def test_validate_topics_private_chat_fails_if_disabled():
    bot = MockBot(
        user=User(id=1, has_topics_enabled=False),
        chats={100: Chat(id=100, type="private")},
        members={}
    )
    settings = TelegramTopicsSettings(enabled=True, scope="main")
    with pytest.raises(ConfigError, match="topics enabled for private chat"):
        await _validate_topics_setup_for(
            bot=bot,
            topics=settings,
            chat_id=100,
            project_chat_ids=[]
        )

@pytest.mark.anyio
async def test_validate_topics_supergroup_success():
    bot = MockBot(
        user=User(id=1),
        chats={100: Chat(id=100, type="supergroup", is_forum=True)},
        members={(100, 1): ChatMember(status="administrator", can_manage_topics=True)}
    )
    settings = TelegramTopicsSettings(enabled=True, scope="main")
    # should not raise
    await _validate_topics_setup_for(
        bot=bot,
        topics=settings,
        chat_id=100,
        project_chat_ids=[]
    )

