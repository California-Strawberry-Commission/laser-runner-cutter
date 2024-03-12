import React, {
  ChangeEvent,
  FocusEvent,
  useContext,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

import { cn } from "@/lib/utils";
import KeyboardContext from "@/lib/keyboard-context";

export interface InputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange"> {
  onChange?: (value: string) => void;
}

/**
 * This is an internally controlled Input component that is needed to support
 * the virtual keyboard. There are multiple sources of input that this component
 * needs to support: physical keyboard, virtual keyboard, and any external
 * component (via props.value).
 *
 * Physical keyboard inputs are handled via the internal <input>'s onChange. The new
 * value needs to be propagated to <Input>'s state, the virtual keyboard state, and
 * through props.onChange.
 *
 * Virtual keyboard inputs are handled via the function passed into setKeyboardOnChange.
 * Within this function, we need to propagate the new value to <Input>'s state
 * and through props.onChange.
 *
 * External value changes incoming via props.value needs to be propagated to
 * <Input>'s state and the virtual keyboard state. This is achieved in the effect hook.
 */
const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, value: propValue, onFocus, onChange, ...props }, ref) => {
    const innerRef = useRef<HTMLInputElement>(null);
    const [value, setValue] = useState<string>(
      propValue !== undefined ? String(propValue) : ""
    );
    const { setKeyboardValue, setKeyboardOnChange, setKeyboardVisible } =
      useContext(KeyboardContext);

    useImperativeHandle(ref, () => innerRef.current!, []);

    useEffect(() => {
      const newPropValue = propValue !== undefined ? String(propValue) : "";
      setValue(newPropValue);
      setKeyboardValue(newPropValue);
    }, [propValue]);

    const onFocusInternal = (e: FocusEvent<HTMLInputElement>) => {
      onFocus && onFocus(e);
      const currentValue = innerRef.current ? innerRef.current.value : "";
      setKeyboardValue(currentValue);
      setKeyboardOnChange((value: string) => {
        setValue(value);
        onChange && onChange(value);
      });
      setKeyboardVisible(true);
    };

    const onChangeInternal = (e: ChangeEvent<HTMLInputElement>) => {
      const newValue = e.target.value;
      setValue(newValue);
      setKeyboardValue(newValue);
      onChange && onChange(newValue);
    };

    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        ref={innerRef}
        value={value}
        onChange={onChangeInternal}
        onFocus={onFocusInternal}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
