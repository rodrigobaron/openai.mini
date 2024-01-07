import MessageComp from "./message";

import type { Message, Plugin } from "./types";

type MessageListProps = {
  messages: Message[];
  avatars: { user: string | React.ReactNode; assistant: string | React.ReactNode };
};

const MessageList = ({ messages, avatars }: MessageListProps) => {
  return (
    <>
      {messages?.map((message, index) => {
        // let plugin = undefined;

        // if (message.role === "assistant") {
        //   plugin = {
        //     name: "WebPilot",
        //     status: "done" as any,
        //     request: dummyRequest,
        //     response: dummyResponse,
        //   } as Plugin;
        // }

        return <MessageComp key={index} role={message?.role} content={message?.content} plugin={message.plugin} avatars={avatars} />;
      })}
    </>
  );
};

export default MessageList;

const dummyRequest = `
{
  "link": "https://www.google.com/search?q=Weather on August 9, 2022",
  "lp": false,
  "ur": "Weather on August 9, 2022",
  "l": "zh-CN",
  "rt": false
}
`;

const dummyResponse = `
{
  "title": "",
  "content": "The weather is hot and humid this week - Fushun Municipal People's Government",
}
`;
