import React from "react";

import { Input, InputProps } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export interface InputWithLabelProps extends InputProps {
  label?: string;
  helper_text?: string;
}

const InputWithLabel = React.forwardRef<HTMLInputElement, InputWithLabelProps>(
  ({ label, helper_text, ...props }, ref) => {
    return (
      <div className="relative">
        <Input {...props} ref={ref} />
        {label && (
          <Label className="absolute left-[8px] top-[-8px] text-xs bg-background pointer-events-none text-nowrap">
            {label}
          </Label>
        )}
        {helper_text && (
          <p className="absolute left-[8px] bottom-[-18px] text-xs pointer-events-none text-nowrap text-muted-foreground">
            {helper_text}
          </p>
        )}
      </div>
    );
  }
);
InputWithLabel.displayName = "InputWithLabel";

export { InputWithLabel };
