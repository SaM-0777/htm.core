export interface SelectProps {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}

export function Select({ label, value, options, onChange }: SelectProps) {
  return (
    <div className="flex-1">
      <label className="block text-[8px] uppercase tracking-widest text-slate-500 mb-2">
        {label}
      </label>
      <div className="relative">
        <select
          value={value}
          onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
            onChange(e.target.value)
          }
          className="w-full bg-transparent border-b border-white/10 text-white font-mono text-xs pb-1 outline-none appearance-none cursor-pointer hover:border-white/30 transition-colors focus:border-white focus:bg-black"
        >
          {options.map((opt) => (
            <option
              key={opt}
              value={opt}
              className="bg-[#050505] text-white py-2"
            >
              {opt}
            </option>
          ))}
        </select>
        <div className="absolute right-0 top-1/2 -translate-y-1/2 pointer-events-none text-white/50 text-[8px]">
          ▼
        </div>
      </div>
    </div>
  );
}
