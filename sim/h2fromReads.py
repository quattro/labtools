#! /usr/bin/env python
import argparse as ap
import math
import os
import sys

import numpy as np
from scipy import stats as sts

import reml


def fformat(x):
    return "{:.6f}".format(x)


def main(args):
    argp = ap.ArgumentParser(description="Generate Likelihoods from a Genotype.")
    argp.add_argument("geno", type=ap.FileType("r"), help="Genotype numpy matrix.")
    argp.add_argument("pheno", type=ap.FileType("r"), help="Phenotype")
    argp.add_argument("cov", type=float, help="The mean coverage amount.")
    argp.add_argument("-m", "--method", choices=["EM", "REML"], default="REML")
    argp.add_argument("-s", "--use_sample_var", action="store_true", default=False)
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-e", "--errorrate", type=float, help="Sequencing error rate.",
                      default=0.01)
    argp.add_argument("-o", "--output", type=ap.FileType("w"),
                      default=sys.stdout)

    args = argp.parse_args(args)
    geno = np.load(args.geno)

    pheno = np.loadtxt(args.pheno, dtype=str)
    pheno = np.array(pheno.T[2], dtype=float)

    n, m = geno.shape

    likes = np.zeros((3, m))
    g = np.array([np.arange(3) for _ in range(m)])

    freqs = np.mean(geno, axis=0) / 2.0
    probs = np.zeros((3, m))

    probs[0] = (1 - freqs) ** 2
    probs[1] = 2 * (1 - freqs) * freqs
    probs[2] = freqs ** 2

    # compute coverage per person per snp
    rows = []
    for idx in range(n):
        sgeno = geno[idx]

        covs = np.random.poisson(args.cov, size=m)

        # older version of scipy has bug if you pass 0... do this for workaround
        num1s = np.zeros(m)
        num1s[covs != 0] = sts.binom.rvs(covs[covs != 0], 1 - args.errorrate)

        # flip num1s for homozygous minor case
        mask = sgeno == 0
        num1s[mask] = covs[mask] - num1s[mask]

        # simulate reads for heterozygous case where coverage is positive
        mask = np.logical_and(sgeno == 1, covs != 0)
        num1s[mask] = sts.binom.rvs(covs[mask], 0.5)

        num0s = covs - num1s

        likes[0][:] = (math.log(args.errorrate) * num1s) + (math.log(1 - args.errorrate) * num0s)
        likes[1][:] = math.log(0.5) * covs
        likes[2][:] = (math.log(args.errorrate) * num0s) + (math.log(1 - args.errorrate) * num1s)

        # convert to likelihoods
        likes = np.exp(likes)
        likes = likes / np.sum(likes, axis=0)

        # get expected-genotype dosages
        #row = np.sum(likes * probs * g.T, axis=0) / np.sum(likes * probs, axis=0)
        row = np.sum(likes * g.T, axis=0) / np.sum(likes, axis=0)
        rows.append(row)

    d = np.array(rows)
    # standardize genotype
    #z = (d - 2 * freqs) / np.sqrt(2 * freqs * (1 - freqs))
    z = (d - 2 * freqs) / np.std(d, axis=0)


    # create GRM
    w = (1 / float(m)) * z.dot(z.T)

    #import pdb; pdb.set_trace()
    initial = np.array([.5, .5])
    if args.method == "EM":
        h2g = reml.emREML(w, pheno, initial, X=None, calc_se=True, max_iter=500, verbose=args.verbose)
    elif args.method == "REML":
        h2g = reml.aiREML(w, pheno, initial, X=None, calc_se=True, max_iter=500, verbose=args.verbose)

    var, se, s = h2g
    total = sum(var)
    args.output.write("\t".join(["Source", "Variance", "SE"]) + os.linesep)
    args.output.write("\t".join(["V(G)", fformat(var[0]), fformat(math.sqrt(s[0, 0]))]) + os.linesep)
    args.output.write("\t".join(["V(e)", fformat(var[1]), fformat(math.sqrt(s[1, 1]))]) + os.linesep)
    args.output.write("\t".join(["V(G)/Vp", fformat(var[0] / total), fformat(se[0])]) + os.linesep)
    args.output.write("Variance/Covariance Matrix" + os.linesep)
    args.output.write(str(s) + os.linesep)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
