import { cn } from "@/utils/tailwind-merge";
import { HTMLAttributes } from "react";

export interface InputProps {
  value: number | "";
  onChange: (value: number | "") => void;
  min?: number;
  max?: number;
  placeholder?: string;
  className: HTMLAttributes<HTMLInputElement>["className"];
}

export function Input({
  value,
  onChange,
  min,
  max,
  placeholder,
  className = "w-12",
}: InputProps) {
  return (
    <input
      type="number"
      min={min}
      max={max}
      placeholder={placeholder}
      value={value}
      onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
        onChange(e.target.value ? parseInt(e.target.value) : "")
      }
      className={cn(
        `bg-transparent border-b border-white/10 text-white font-mono text-sm outline-none text-center pb-1 focus:border-white transition-colors placeholder:text-white/20`,
        className,
      )}
    />
  );
}
