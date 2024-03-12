import { useEffect, useRef, useState } from "react";
import Keyboard, { KeyboardReactInterface } from "react-simple-keyboard";
import "react-simple-keyboard/build/css/index.css";
import KeyboardContext, { KeyboardOnChange } from "@/lib/keyboard-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function App() {
  const [keyboardOnChange, setKeyboardOnChange] = useState<KeyboardOnChange>();
  const [keyboardVisible, setKeyboardVisible] = useState<boolean>(false);
  const [myString, setMyString] = useState<string>("Initial value");
  const [myNumber, setMyNumber] = useState<number>(0);
  const [layoutName, setLayoutName] = useState("default");
  const keyboard = useRef<KeyboardReactInterface>();

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
    return window.removeEventListener("click", clickHandler, true);
  }, []);

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
          id="myStringInput"
          name="myStringInput"
          type="text"
          value={myString}
          placeholder={"Tap on the virtual keyboard to start"}
          onChange={(value) => {
            setMyString(value);
          }}
        />
        <Input
          id="myNumber"
          name="myNumber"
          type="number"
          value={myNumber}
          placeholder={"Tap on the virtual keyboard to start"}
          onChange={(value) => {
            const newValue = Number(value);
            setMyNumber(isNaN(newValue) ? 0 : newValue);
          }}
        />
        <Button
          onClick={() => {
            const newValue = Math.floor(Math.random() * 100);
            setMyNumber(newValue);
          }}
        >
          Random number above
        </Button>
      </KeyboardContext.Provider>
      {/* Render as hidden because we need the keyboard ref at all times */}
      <div
        className={`fixed bottom-0 w-full flex justify-center z-10 ${keyboardVisible ? "" : "hidden"}`}
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
    </div>
  );
}
