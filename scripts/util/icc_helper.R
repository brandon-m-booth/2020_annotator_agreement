#!/usr/bin/env Rscript
library("optparse");
library("irr");

option_list = list(
  make_option(c("-i", "--input_csv"), type="character", help="Path to csv containing the signals (rows are time, columns are signals)"),
  make_option(c("-o", "--output_csv"), type="character", help="Path to output csv containing ICC results")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$input_csv)) {
  print_help(opt_parser)
  stop("An input argument must be supplied", call.=FALSE)
}

if (is.null(opt$output_csv)) {
  print_help(opt_parser)
  stop("An output argument must be supplied", call.=FALSE)
}

df = read.table(opt$input_csv, sep=",", header=TRUE);
icc_results = icc(df, model="oneway", type="agreement", unit="average")
write.csv(icc_results$value, opt$output_csv);
