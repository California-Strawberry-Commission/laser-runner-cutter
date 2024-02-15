import { createContext } from "react";

export type KeyboardOnChange = (value: string) => void;

const KeyboardContext: React.Context<{
  setKeyboardValue: (value: string) => void;
  setKeyboardOnChange: (onChange: KeyboardOnChange) => void;
  setKeyboardVisible: (visible: boolean) => void;
}> = createContext({
  setKeyboardValue: (_) => {},
  setKeyboardOnChange: (_) => {},
  setKeyboardVisible: (_) => {},
});

export default KeyboardContext;
