"use client";

import "./App.css";
import { useState } from "react";
import { FileUpload } from "@/components/ui/file-upload.tsx";

type ButtonStatusStates = "blocked" | "unblocked" | "generating";

function App() {
  // console.log(import.meta.env.VITE_API_URL)
  const [image, setImage] = useState<FormData>();
  const [buttonStatus, setButtonStatus] = useState<ButtonStatusStates>("blocked");
  const [captionStatus, setCaptionStatus] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>("Caption will be shown here");

  const capitalize = (str: string) =>
    str ? str[0].toUpperCase() + str.slice(1) : "";

  const handleFileUpload = async (files: File[]) => {
    const formData = new FormData();
    formData.append("file", files[0]);
    setImage(formData);
    setButtonStatus("unblocked");
  };

  const generateCaption = async () => {
    setButtonStatus("generating");
    const res = await fetch(`${import.meta.env.VITE_API_URL}/generate-caption`, {
      method: "POST",
      headers: {
        accept: "application/json",
      },
      body: image,
    });

    const text = await res.json();
    
    setCaption(capitalize(text['caption']));
    setButtonStatus("blocked");
    setCaptionStatus(true);
  };

  return (
    <main className="min-h-screen w-full bg-black text-gray-500 p-5 flex flex-col justify-center items-center">
      <h1 className="h-15 mb- text-5xl font-bold bg-linear-to-r from-red-500 to-violet-700 bg-clip-text text-transparent font-michroma">
        Caption Your Images
      </h1>
      <div className="h-[30rem] w-[40rem] border m-10 rounded-2xl flex flex-col gap-5">
        <div className="">
          <FileUpload onChange={handleFileUpload} />
        </div>
        <div className="flex justify-center h-8">
          <button
            onClick={() => generateCaption()}
            className={`${buttonStatus == "blocked" || buttonStatus === "generating" ? 'text-neutral-500' : 'text-neutral-300'} bg-neutral-800 border-gray-800 px-4 rounded-sm`}
            disabled={buttonStatus === "blocked" || buttonStatus === "generating" ? true : false}
          >
            {buttonStatus === 'generating' ? 'Generating...' : 'Generate'}
          </button>
        </div>
        <div className={`text-xl h-25 w-full px-7 pt-4 rounded-xl ${captionStatus ? 'text-neutral-300': 'text-neutral-600'}`}>
          {caption}
        </div>
      </div>
    </main>
  );
}

export default App;
