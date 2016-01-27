#! /usr/bin/env python
import argparse as ap
import gzip
import os
import struct
import sys

import numpy as np
import numpy.linalg as linalg
import scipy.optimize as opt


# CONSTANTS
FORMAT = 8
G = np.arange(3)

MAF = "MAF"
ENC = "ENC"
VAR = "VAR"
SCALE = "SCALE"

MAF_R = "READS"
MAF_G = "GENOTYPE"
MAFS = [MAF_R, MAF_G]

ENC_D = "DOSAGE"
ENC_B = "BESTGUESS"
ENCS = [ENC_D, ENC_B]

VAR_S = "SAMPLE"
VAR_E = "EXPECTED"
VARS = [VAR_S, VAR_E]

SCALE_P = "PAIRWISE"
SCALE_M = "M"
SCALES = [SCALE_P, SCALE_M]


def get_posteriors(row, probs, args):
    n, p = probs.shape

    format = row[FORMAT].split(":")
    glidx = -1
    try:
        glidx = format.index("GL")
    except ValueError:
        # if likelihoods aren't available not much we can do
        return

    miss = np.zeros(n, dtype=bool)
    for idx, entry in enumerate(row[FORMAT + 1:]):
        # get the log10-scaled likelihoods
        likes = np.power(10, map(float, entry.split(":")[glidx].split(",")))
        miss[idx] = sum(likes) == 3.0

        # convert to posterior probabilities using reference prior
        probs[idx][0] = likes[0] 
        probs[idx][1] = likes[1] 
        probs[idx][2] = likes[2] 

    # estimate the allele frequency by optimizing the likelihood of the data wrt to f
    def nll(f):
        fs = np.array([(1 - f) ** 2, 2 * f * (1 - f), f ** 2])
        # this sums the negative log-likelihood for each sample
        # NLL(D|f) =  - sum_i log( sum_g Pr(Reads_i, G_i = g) * Pr(g | f) )
        val = -np.log((probs * fs.T).sum(axis=1)).sum()
        return val

    if args.maf == MAF_R:
        # estimate MAF directly from reads
        vals = [slice(0.01, 0.49, 0.01)]
        f = opt.brute(nll, ranges=vals)
        fs = np.array([(1 - f) ** 2, 2 * f * (1 - f), f ** 2])
        # convert likelihoods to posterior probabilities
        probs = ((probs * fs).T / (probs * fs).sum(axis=1)).T
        maf = f
        # get posterior mean = dosage
        dose = np.array(map(lambda x: sum(G * x), probs))

        if args.encoding == ENC_B:
            # round to best-guess genotype
            dose = np.round(dose)
    else:
        # assume flat prior & scale likelihoods
        probs = (probs.T / probs.sum(axis=1)).T

        # get posterior mean = dosage
        dose = np.array(map(lambda x: sum(G * x), probs))
        if args.encoding == ENC_D:
            # estimate MAF from genotypes
            maf = np.mean(dose) / 2.0
        else:
            # round to best-guess genotype
            dose = np.round(dose)
            maf = np.mean(dose) / 2.0

    return dose, maf, miss


def parse_line(row, probs, args):
    dose, maf, miss = get_posteriors(row, probs, args)

    not_miss = np.logical_not(miss)
    dose[not_miss] -= 2 * maf

    if args.variance == VAR_E:
        dose[not_miss] /= np.sqrt(2 * maf * (1 - maf))
    else:
        dose[not_miss] /= np.std(dose[not_miss], ddof=1)

    dose[miss] = 0.0

    return dose, not_miss


def calc_sub_grm(rows, covs, args):
    Z = np.array(rows)
    C = np.array(covs).astype(float)

    m, n = Z.shape
    Atmp = Z.T.dot(Z)

    if args.scale == SCALE_P:
        Mtmp = C.T.dot(C)
    else:
        Mtmp = m * np.ones((n,n))

    Atmp /= Mtmp
    Atmp[np.isnan(Atmp)] = 0.0
    return Atmp, Mtmp


def calc_grm(vcf, args):
    rows = []
    covs = []
    for line in vcf:
        # skip metadata
        if "##" in line:
            continue

        # get header
        if "#CHROM" in line:
            ids = line.split()[FORMAT + 1:]
            n = len(ids)
            probs = np.zeros((n, 3))
            A = np.zeros((n,n))
            M = np.zeros((n,n))
            continue

        row = line.split()

        chr = row[0]
        pos = row[1]
        key = row[2]

        row, cov = parse_line(row, probs, args)
        rows.append(row)
        covs.append(cov)
        if len(row) == args.max_snp:
            Atmp, Mtmp = calc_sub_grm(rows, covs, args)
            A += Atmp
            M += Mtmp
            rows[:] = []
            covs[:] = []

    if len(row) > 0:
        Atmp, Mtmp = calc_sub_grm(rows, covs, args)
        A += Atmp
        M += Mtmp


    return A, M, ids


def write_grm(prefix, A, M, ids):
    n, n = A.shape
    with open("{}.grm.bin".format(prefix), "wb") as grmfile:
        for idx in range(n):
            for jdx in range(idx + 1):
                val = struct.pack('f', A[idx, jdx])
                grmfile.write(val)

    with open("{}.grm.N.bin".format(prefix), "wb") as grmfile:
        for idx in range(n):
            for jdx in range(idx + 1):
                val = struct.pack('f', M[idx, jdx])
                grmfile.write(val)

    with open("{}.grm.id".format(prefix), "w") as grmfile:
        for idx in range(n):
            fid = ids[idx]
            iid = ids[idx].replace("F", "I")
            grmfile.write("\t".join([fid, iid]) + os.linesep)

    return


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("vcf_file")
    argp.add_argument("prefix")
    argp.add_argument("-e", "--encoding", choices=ENCS, default=ENC_D, help="Encoding of genotype.")
    argp.add_argument("-v", "--variance", choices=VARS, default=VAR_S, help="Standardize by sample or expected variance.")
    argp.add_argument("-f", "--maf", choices=MAFS, default=MAF_R, help="Calculate MAF from Reads or Genotype.")
    argp.add_argument("-s", "--scale", choices=SCALES, default=SCALE_P, help="Scale GRM by pair-wise # of SNPs or all.")
    argp.add_argument("-m", "--max-snp", type=int, default=50000, help="Window size to use when constructing GRM. Saves memory.")

    args = argp.parse_args(args)


    with gzip.open(args.vcf_file, "r") as vcf:
        A, M, ids = calc_grm(vcf, args)
        write_grm(args.prefix, A, M, ids)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
