import React, { useEffect, useRef, useState } from "react";
import Keyboard from "react-simple-keyboard";
import CameraPreview from "@/components/camera-preview";
import ButtonBar from "@/components/button-bar";
import "react-simple-keyboard/build/css/index.css";
import KeyboardContext from "@/lib/keyboard-context";

const DEFAULT_INPUTS = {
  saveDir: "~/Pictures/runners",
  filePrefix: "runner_",
  exposure: "0.2",
  interval: "5",
};

function App() {
  const [inputs, setInputs] = useState(DEFAULT_INPUTS);
  const [layoutName, setLayoutName] = useState("default");
  const [inputName, setInputName] = useState("default");
  const keyboard = useRef<React.ElementRef<typeof Keyboard>>();

  useEffect(() => {
    if (keyboard.current != null) {
      // TODO: set initial values on keyboard via keyboard.current.setInput
    }
  }, []);

  const onChangeAll = (inputs: any) => {
    // Spread the inputs into a new object so that we trigger a re-render
    setInputs({ ...inputs });
    console.log("Inputs changed", inputs);
  };

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
        value={{ inputs, setInputs, layoutName, setInputName }}
      >
        <main className="flex flex-col gap-4">
          <div className="flex flex-row gap-4">
            <div>
              <h1 className="text-3xl font-bold pb-4">Runner Image Capture</h1>
              <ButtonBar />
            </div>
            <CameraPreview />
          </div>
          <div className="w-[800px]">
            <Keyboard
              keyboardRef={(r) => (keyboard.current = r)}
              inputName={inputName}
              layoutName={layoutName}
              onChangeAll={onChangeAll}
              onKeyPress={onKeyPress}
            />
          </div>
        </main>
      </KeyboardContext.Provider>
    </div>
  );
}

export default App;
