import { useState } from 'react'
import './App.css'
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { MainContainer, ChatContainer, MessageList, Message, MessageInput, TypingIndicator } from '@chatscope/chat-ui-kit-react';
import userprofile from './assets/user-profile.jpg'

function App() {
  const [messages, setMessages] = useState([
    {
      message: "Hello, I'm Bot! Ask me anything!",
      sentTime: "just now",
      sender: "Bot"
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (message) => {
    const newMessage = {
      message,
      direction: 'outgoing',
      sender: "user"
    };
    const newMessages = [...messages, newMessage];
    setMessages(newMessages);

    setIsTyping(true);
    await processMessage(newMessage);
  };

  async function processMessage(chatMessages) { 

    console.log('chatMessage', chatMessages) // TODO: You can watch what messages is there.

   // TODO: change url

    await fetch("http://127.0.0.1:8000/generate-response", 
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({"query": chatMessages?.message}) // send data from frontend
    }).then((data) => {
      return data.json();
    }).then((data) => {
      console.log('final',data);   // TODO : you can see what data is coming from backend
      setMessages([...messages,  {
        message: data.question,
        direction: 'outgoing',
        sender: "user"
      }, {
        message: data?.response, 
        sender: "Bot"
      }]);
      setIsTyping(false);
    });
  }
  console.log('output', messages)

  return (
    <div>
      <div className='container'>
        <div>
          <h2 className='heading'>Welcome to  Agentic RAG Chat</h2>

        </div>
        <div className='profileContainer'>
          <img className='profile' src={userprofile} width='70px' height="70px" />
        </div>
      </div>
    <div className="App">
      <div style={{ position:"relative", height: "80vh", width: "85vw"  }}>
        <MainContainer>
          <ChatContainer>       
            <MessageList className='inBoxMessages'
              scrollBehavior="smooth" 
              typingIndicator={isTyping ? <TypingIndicator content="Bot is typing" /> : null}
            >
              {messages.map((message, i) => {
                console.log(message)
                return <Message key={i} model={message} />
              })}
            </MessageList>
            <MessageInput placeholder="Type message here" onSend={handleSend} />        
          </ChatContainer>
        </MainContainer>
      </div>
    </div>
    </div>

  )
}

export default App
