import React, {
  FocusEvent,
  useContext,
  useImperativeHandle,
  useRef,
} from "react";

import { cn } from "@/lib/utils";
import KeyboardContext, { KeyboardOnChange } from "@/lib/keyboard-context";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  keyboardOnChange?: KeyboardOnChange;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  // TODO: this currently can only be used as a controlled component
  ({ className, type, onFocus, onBlur, keyboardOnChange, ...props }, ref) => {
    const innerRef = useRef<HTMLInputElement>(null);
    const { setKeyboardValue, setKeyboardOnChange, setKeyboardVisible } =
      useContext(KeyboardContext);

    useImperativeHandle(ref, () => innerRef.current!, []);

    const onFocusInternal = (e: FocusEvent<HTMLInputElement>) => {
      onFocus && onFocus(e);
      const currentValue = innerRef.current ? innerRef.current.value : "";
      setKeyboardValue(currentValue);
      setKeyboardOnChange((value: string) => {
        keyboardOnChange && keyboardOnChange(value);
      });
      setKeyboardVisible(true);
    };

    const onBlurInternal = (e: FocusEvent<HTMLInputElement>) => {
      onBlur && onBlur(e);
      // TODO: onBlur is triggered when keyboard is clicked
      // setKeyboardVisible(false);
    };

    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={innerRef}
        onFocus={onFocusInternal}
        onBlur={onBlurInternal}
        // Prevent physical keyboard input
        onKeyDown={(event) => {
          event.preventDefault();
        }}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
