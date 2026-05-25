export interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function Toggle({ label, checked, onChange }: ToggleProps) {
  return (
    <div
      className="flex items-center gap-3 cursor-pointer group w-max"
      onClick={() => onChange(!checked)}
    >
      <div
        className={`w-3 h-3 rounded-[1px] border flex items-center justify-center transition-all duration-300 ${checked ? "border-white bg-white shadow-[0_0_8px_rgba(255,255,255,0.6)]" : "border-white/30 bg-transparent"}`}
      >
        {checked && <div className="w-1.5 h-1.5 bg-black" />}
      </div>
      <span
        className={`text-[9px] uppercase tracking-widest transition-colors ${checked ? "text-white" : "text-slate-500 group-hover:text-slate-400"}`}
      >
        {label}
      </span>
    </div>
  );
}
