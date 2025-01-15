"use client";

import { useEffect, useRef, useState } from "react";
import Keyboard, { KeyboardReactInterface } from "react-simple-keyboard";
import "react-simple-keyboard/build/css/index.css";
import { cn } from "@/lib/utils";
import KeyboardContext, {
  KeyboardOnChange,
} from "@/lib/keyboard/KeyboardContext";

export default function KeyboardProvider({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const [keyboardOnChange, setKeyboardOnChange] = useState<KeyboardOnChange>();
  const [keyboardVisible, setKeyboardVisible] = useState<boolean>(false);
  const [layoutName, setLayoutName] = useState("default");
  const keyboard = useRef<KeyboardReactInterface>(null);

  // Add a global click listener to dismiss the virtual keyboard if user clicks outside
  // of it
  useEffect(() => {
    function clickHandler(e: any) {
      const clickedInput = e.target.nodeName === "INPUT";
      let clickedVirtualKeyboard = false;
      // Check if the clicked element or any of its ancestors have the class name "react-simple-keyboard"
      let element = e.target;
      while (element && element !== document.body && element.classList) {
        if (element.classList.contains("react-simple-keyboard")) {
          clickedVirtualKeyboard = true;
          break;
        }
        // Move to the parent node of the current element
        element = element.parentNode;
      }

      if (!clickedInput && !clickedVirtualKeyboard) {
        setKeyboardVisible(false);
      }
    }

    window.addEventListener("click", clickHandler);
    return () => {
      window.removeEventListener("click", clickHandler, true);
    };
  }, []);

  const handleShift = () => {
    const newLayoutName = layoutName === "default" ? "shift" : "default";
    setLayoutName(newLayoutName);
  };

  const onKeyPress = (button: string) => {
    if (button === "{shift}" || button === "{lock}") {
      handleShift();
    }
  };

  return (
    <>
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
        {children}
      </KeyboardContext.Provider>
      {/* Render as hidden because we need the keyboard ref at all times */}
      <div
        className={cn(
          "fixed bottom-0 w-full flex justify-center z-10",
          keyboardVisible ? "" : "hidden"
        )}
      >
        <div className="w-[800px]">
          <Keyboard
            keyboardRef={(r) => (keyboard.current = r)}
            layoutName={layoutName}
            onChange={keyboardOnChange}
            onKeyPress={onKeyPress}
          />
        </div>
      </div>
    </>
  );
}
