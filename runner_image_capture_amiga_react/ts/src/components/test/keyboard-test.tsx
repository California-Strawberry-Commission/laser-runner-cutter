import React, { useRef, useState } from "react";
import Keyboard, { KeyboardReactInterface } from "react-simple-keyboard";
import "react-simple-keyboard/build/css/index.css";
import KeyboardContext, { KeyboardOnChange } from "@/lib/keyboard-context";
import { Input } from "@/components/ui/input";

export default function App() {
  const [keyboardOnChange, setKeyboardOnChange] = useState<KeyboardOnChange>();
  const [keyboardVisible, setKeyboardVisible] = useState<boolean>(false);
  const [saveDir, setSaveDir] = useState("InitialSaveDir");
  const [filePrefix, setFilePrefix] = useState("InitialFilePrefix");
  const [layoutName, setLayoutName] = useState("default");
  const keyboard = useRef<KeyboardReactInterface>();

  const handleShift = () => {
    const newLayoutName = layoutName === "default" ? "shift" : "default";
    setLayoutName(newLayoutName);
  };

  const onKeyPress = (button: string) => {
    console.log("Button pressed", button);
    if (button === "{shift}" || button === "{lock}") {
      handleShift();
    }
  };

  const onChangeFilePrefix = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFilePrefix(e.target.value);
  };

  return (
    <div className="p-4">
      <KeyboardContext.Provider
        value={{
          setKeyboardValue: (value: string) => {
            keyboard.current?.setInput(value);
          },
          setKeyboardOnChange: (onChange: KeyboardOnChange) => {
            setKeyboardOnChange(() => onChange);
          },
          setKeyboardVisible,
        }}
      >
        <Input
          id="saveDir"
          name="saveDir"
          value={saveDir}
          placeholder={"Tap on the virtual keyboard to start"}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
            setSaveDir(e.target.value);
          }}
          keyboardOnChange={(value) => setSaveDir(value)}
        />
        <Input
          id="filePrefix"
          name="filePrefix"
          value={filePrefix}
          placeholder={"Tap on the virtual keyboard to start"}
          onChange={onChangeFilePrefix}
        />
      </KeyboardContext.Provider>
      {keyboardVisible && (
        <div className="fixed bottom-0 w-full flex justify-center">
          <div className="w-[800px]">
            <Keyboard
              keyboardRef={(r) => (keyboard.current = r)}
              layoutName={layoutName}
              onChange={keyboardOnChange}
              onKeyPress={onKeyPress}
            />
          </div>
        </div>
      )}
    </div>
  );
}
