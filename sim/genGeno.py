#! /usr/bin/env python
import argparse as ap
import gzip
import math
import os
import random as rdm
import struct
import sys

import numpy as np

import reml


def fformat(x):
    return "{:.6f}".format(x)


def main(args):
    argp = ap.ArgumentParser(description="Generate a random genotype")
    argp.add_argument("n", type=int, help="Pop number.")
    argp.add_argument("m", type=int, help="Genome length.")
    argp.add_argument("f", type=float, help="Ref allele freq")
    argp.add_argument("h2", type=float, help="Narrow-sense Heritability")
    argp.add_argument("prefix", help="Output prefix")
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout,
                      help="Where to write numpy matrix")
    argp.add_argument("-o2", "--output2", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)
    g = np.random.binomial(2, args.f, size=(args.n, args.m))
    n, m = g.shape

    np.save(args.output, g)

    g = g.astype(float)
    f = np.mean(g, axis=0) / 2

    # standardize the genotype
    #z = (g - (2 * f)) / np.sqrt(2 * f * (1 - f))
    v = np.var(g, axis=0)

    z = (g - (2 * f)) / np.sqrt(v)

    # sample effects
    betas = np.random.normal(0, math.sqrt(args.h2 / float(m)), m)

    # compute sample variances for betas and noise given h2
    g = z.dot(betas)

    s2g = np.var(g)
    s2e = s2g * ( (1.0 / args.h2) - 1 )

    # create phenotypes
    e = np.random.normal(0, math.sqrt(s2e), n)
    y = g + e

    # standardize
    y = (y - np.mean(y)) / np.std(y)

    # output phenotype mapping
    with open("{}.phen".format(args.prefix), "w") as phenfile:
        for idx, p in enumerate(y):
            fid = "FID{}".format(idx)
            iid = "IID{}".format(idx)
            phenfile.write("{} {} {}{}".format(fid, iid, p, os.linesep))

    # compute GRM
    w = (1.0 / float(m)) * z.dot(z.T)

    # output GRM files in GCTA bin format
    with open("{}.grm.bin".format(args.prefix), "wb") as grmfile:
        for idx in range(n):
            for jdx in range(idx + 1):
                val = struct.pack('f', w[idx, jdx])
                grmfile.write(val)

    val = struct.pack('i', int(m))
    with open("{}.grm.N.bin".format(args.prefix), "wb") as grmfile:
        for idx in range(n):
            for jdx in range(idx + 1):
                grmfile.write(val)

    with open("{}.grm.id".format(args.prefix), "w") as grmfile:
        for idx in range(n):
            fid = "FID{}".format(idx)
            iid = "IID{}".format(idx)
            grmfile.write("\t".join([fid, iid]) + os.linesep)

    # compute h2g estimates with AI-REML GCTA-style
    initial = np.array([.5, .5])
    h2g = reml.aiREML(w, y, initial, X=None, calc_se=True, max_iter=500)

    var, se, s = h2g
    total = sum(var)
    args.output2.write("\t".join(["Source", "Variance", "SE"]) + os.linesep)
    args.output2.write("\t".join(["V(G)", fformat(var[0]), fformat(math.sqrt(s[0, 0]))]) + os.linesep)
    args.output2.write("\t".join(["V(e)", fformat(var[1]), fformat(math.sqrt(s[1, 1]))]) + os.linesep)
    args.output2.write("\t".join(["V(G)/Vp", fformat(var[0] / total), fformat(math.sqrt(se[0]))]) + os.linesep)
    args.output2.write("Variance/Covariance Matrix" + os.linesep)
    args.output2.write(str(s) + os.linesep)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
