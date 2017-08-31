library(data.table)
library(logistf)
library(optparse)
library(plink2R)

option_list <- list(
    make_option("--bfile", action="store", default=NA, type='character', help="Path to PLINK binary data [required]"),
    make_option("--pheno", action="store", default=NA, type='character', help="Path to phenotype data [required]"),
    make_option("--covar", action="store", default=NA, type='character', help="Path to covariate data [required]"),
    make_option("--keep", action="store", default=NA, type='character', help="Path to file containing samples to keep for analysis"),
    make_option("--mperm", action="store", default=NA, type='integer', help="Number of permutations to perform"),
    make_option("--quiet", action="store", default=F, type='logical', help="Only output results"),
    make_option("--out", action="store", default="", type='character', help="Path to output files")
)

opts <- optparse::OptionParser(usage = "%prog [options]", option_list=option_list)
opt = optparse::parse_args(opts)
if (is.na(opt$bfile) | is.na(opt$pheno) | is.na(opt$covar) | (!is.na(opt$mperm) & opt$mperm < 1)) {
    optparse::print_help(opts)
    q()
}
if (!opt$quiet) {
    cat("firth.R\n")
    cat("\t--bfile", opt$bfile, "\n")
    cat("\t--pheno", opt$pheno, "\n")
    cat("\t--covar", opt$covar, "\n")
    if (!is.na(opt$keep)) {
        cat("\t--keep", opt$keep, "\n")
    }
    if (!is.na(opt$mperm)) {
        cat("\t--mperm", opt$mperm, "\n")
    }
    if (opt$out != "") {
        cat("\t--out", opt$out, "\n")
    }
}

data = read_plink(opt$bfile, impute="none")
if (!opt$quiet) {
    n <- nrow(data$bed)
    m <- ncol(data$bed)
    cat("Loaded PLINK binary data with", n, "samples and", m, "SNPs.\n")
}

pheno <- fread(opt$pheno)
covar <- fread(opt$covar)

merged <- merge(pheno, covar, by.x="V1", by.y="FID")
merged <- merged[, .(IID, V3, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10)]
if (!opt$quiet) {
    n <- nrow(merged)
    cat("Found", n, "samples in common between phenotype and covariates data.\n")
}

merged <- merged[V3 != -9,]
if (!opt$quiet) {
    nn <- nrow(merged)
    if (nn != n) {
        cat("Dropped ", n - nn, "samples with missing phenotypes.\n")
    }
}

colnames(merged)[2] <- "Pheno"

# set back to traditional 0/1 labels
merged[Pheno == 1,]$Pheno <- 0
merged[Pheno == 2,]$Pheno <- 1

if (!is.na(opt$keep)) {
    keep <- fread(opt$keep, h=F)
    merged <- merged[IID %in% keep$V1,]
    if (!opt$quiet) {
        cat("Loaded", nrow(keep), "samples in keep file.\n")
        cat("Resulted in", nrow(merged), "samples left.\n")
    }
}

m <- match(merged$IID, data$fam$V1)
geno <- data$bed[m,]

smiss <- apply(geno, 1, function(x) mean(is.na(x))) >= 0.1
gmiss <- apply(geno, 2, function(x) mean(is.na(x))) >= 0.1

if (!opt$quiet) {
    cat("Pruned", sum(smiss), "samples for missingness.\n")
    cat("Pruned", sum(gmiss), "SNPs for missingness.\n")
}

merged <- merged[!smiss,]
geno <- geno[!smiss, !gmiss]
bim <- data$bim[!gmiss,]

if (ncol(geno) == 0) {
    if (!opt$quiet) {
        cat("No SNPs left to process!\n")
    }
    q()
}
if (nrow(geno) == 0) {
    if (!opt$quiet) {
        cat("No samples left to process!\n")
    }
    q()
}

N <- ncol(geno)
if (is.na(opt$mperm)) {
    out <- data.frame(CHR=character(N), SNP=character(N), BP=integer(N), A1=character(N), A2=character(N), N=integer(N),
                      OR=numeric(N), SE=numeric(N), L95=numeric(N), U95=numeric(N), P=numeric(N))
} else {
    out <- data.frame(CHR=character(N), SNP=character(N), BP=integer(N), A1=character(N), A2=character(N), N=integer(N),
                      OR=numeric(N), SE=numeric(N), L95=numeric(N), U95=numeric(N), P=numeric(N), EMP.P=numeric(N))
}

for (idx in 1:ncol(geno)) {
    out$CHR <- bim[idx,]$V1
    out$SNP <- bim[idx,]$V2
    out$BP <- bim[idx,]$V4
    out$A1 <- bim[idx,]$V5
    out$A2 <- bim[idx,]$V6

    sub_data <- cbind(merged, SNP=geno[,idx])
    sub_data <- sub_data[!is.na(sub_data$SNP),]

    res <- logistf::logistf(Pheno ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + SNP, data=sub_data)
    r_idx <- length(res$coefficients)
    out$N[idx] <- nrow(sub_data)
    out$OR[idx] <- format(exp(res$coefficients[r_idx]), digits=4)
    out$SE[idx] <- format(sqrt(res$var[r_idx, r_idx]), digits=4)
    out$L95[idx] <- format(exp(res$ci.lower[r_idx]), digits=4)
    out$U95[idx] <- format(exp(res$ci.upper[r_idx]), digits=4)
    out$P[idx] <- format(res$prob[r_idx], digits=4)
    if (!is.na(opt$mperm)) {
        tally <- 1
        pheno <- sub_data$Pheno
        pval <- res$prob[r_idx]
        for (p in 1:opt$mperm) {
            sub_data$Pheno <- sample(pheno)
            res <- logistf::logistf(Pheno ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + SNP, data=sub_data)
            tally <- tally + as.integer(res$prob[r_idx] <= pval)
        }
        out$EMP.P[idx] <- format(tally / (1 + opt$mperm), digits=4)
    }
}
write.table(out[1,], opt$out, row.names=F, quote=F, col.names=T)
