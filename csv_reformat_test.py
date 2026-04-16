from argparse import ArgumentParser
from pathlib import Path
import re

import pandas as pd


def bytes_to_mb(num_bytes: int) -> float:
	return num_bytes / (1024.0 * 1024.0)


def format_size(path: Path) -> str:
	return f"{bytes_to_mb(path.stat().st_size):.2f} MB"


def reduce_numeric_precision(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
	df_out = df.copy()
	numeric_cols = df_out.select_dtypes(include=["number"]).columns
	df_out[numeric_cols] = df_out[numeric_cols].round(decimals)
	return df_out


def _round_numeric_tokens_in_text(text: str, decimals: int) -> str:
	float_re = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)")

	def _repl(match: re.Match) -> str:
		token = match.group(0)
		try:
			value = float(token)
		except ValueError:
			return token
		return f"{value:.{decimals}f}"

	return float_re.sub(_repl, text)


def round_list_like_string_columns(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
	df_out = df.copy()
	object_cols = df_out.select_dtypes(include=["object"]).columns

	for col in object_cols:
		s = df_out[col]
		non_null = s.dropna()
		if non_null.empty:
			continue

		first = str(non_null.iloc[0]).strip()
		# StatTracker stores obs/action/TD_error_list as bracketed list strings in CSV.
		if not first.startswith("["):
			continue

		df_out[col] = s.map(lambda x: _round_numeric_tokens_in_text(x, decimals) if isinstance(x, str) else x)

	return df_out


def write_parquet(df: pd.DataFrame, out_path: Path, compression: str = "zstd") -> None:
	# Explicit engine gives a clearer failure mode if dependency is missing.
	df.to_parquet(out_path, compression=compression, engine="pyarrow")


def write_csv(df: pd.DataFrame, out_path: Path, decimals: int) -> None:
	float_fmt = f"%.{decimals}f"
	df.to_csv(out_path, index=False, float_format=float_fmt)


def main() -> None:
	parser = ArgumentParser(description="Test parquet size with reduced numeric precision.")
	parser.add_argument(
		"csv_path",
		nargs="?",
		default="TrainingLite/rl_racing/models/FINAL_UTD025_250k_A0.6_R0.8_CPTDFalse_CUTrue_OW2.0_HW2.0_RW2.0/stat_logs/stats_log.csv",
		help="Path to input CSV file.",
	)
	parser.add_argument(
		"--decimals",
		nargs="+",
		type=int,
		default=[6, 4, 3, 2],
		help="Decimal places to test for numeric columns.",
	)
	args = parser.parse_args()

	csv_path = Path(args.csv_path)
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	print(f"Loading CSV: {csv_path}")
	df = pd.read_csv(csv_path)
	base_size_bytes = csv_path.stat().st_size

	try:
		base_parquet = csv_path.with_suffix(".parquet")
		write_parquet(df, base_parquet)
		print("\nBaseline parquet (no rounding):")
		print(f"  {base_parquet.name}: {format_size(base_parquet)}")

		print("\nRounded parquet variants:")
		for dec in sorted(set(args.decimals), reverse=True):
			rounded = reduce_numeric_precision(df, decimals=dec)
			rounded = round_list_like_string_columns(rounded, decimals=dec)
			out_path = csv_path.with_name(f"{csv_path.stem}_d{dec}.parquet")
			write_parquet(rounded, out_path)

			out_bytes = out_path.stat().st_size
			vs_csv = 100.0 * (1.0 - (out_bytes / base_size_bytes))
			print(f"  d={dec}: {out_path.name} -> {format_size(out_path)} ({vs_csv:.1f}% smaller than CSV)")

		print("\nRounded CSV variants:")
		for dec in sorted(set(args.decimals), reverse=True):
			rounded = reduce_numeric_precision(df, decimals=dec)
			rounded = round_list_like_string_columns(rounded, decimals=dec)
			out_csv = csv_path.with_name(f"{csv_path.stem}_d{dec}.csv")
			write_csv(rounded, out_csv, decimals=dec)

			out_bytes = out_csv.stat().st_size
			vs_csv = 100.0 * (1.0 - (out_bytes / base_size_bytes))
			print(f"  d={dec}: {out_csv.name} -> {format_size(out_csv)} ({vs_csv:.1f}% smaller than original CSV)")

	except ImportError as exc:
		print("Parquet engine missing. Install one of:")
		print("  conda install -n f1t pyarrow")
		print("  pip install pyarrow")
		print(f"Original error: {exc}")

		gz_out = csv_path.with_suffix(".csv.gz")
		df.to_csv(gz_out, index=False, compression="gzip")
		print(f"Saved gzip CSV fallback: {gz_out}")


if __name__ == "__main__":
	main()