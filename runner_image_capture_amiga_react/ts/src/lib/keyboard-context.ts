import { createContext } from "react";

const KeyboardContext = createContext({
  inputs: {},
  setInputs: (inputs: {}) => {},
  layoutName: "default",
  setInputName: (inputName: string) => {},
});

export default KeyboardContext;
