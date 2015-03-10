#! /usr/bin/env python
#
# This is used to generate effects under the Eyre-Walker model
#
# alpha_i = delta * S^tau * (1 + eps)
# where alpha_i is the effect of the ith SNP
# delta in {-1, 1}
# S is 4 * N_e * s where s is the selection value
# tau is a real value that indicates the amount of relationship between S and alpha
# eps is ~ N(0, var_eps)

import argparse as ap
import math
import os
import random as rdm
import sys


def get_scores(sscores):
    scores = dict()
    for line in sscores:
        row = line.split()
        scores[row[0]] = float(row[1])

    return scores

def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("snps", type=ap.FileType("r"))
    argp.add_argument("sscores", type=ap.FileType("r"))
    argp.add_argument("--h2", type=float, default=0.33)
    argp.add_argument("-t", "--tau", type=float, default=0.0)
    argp.add_argument("-estd", "--epsstd", type=float, default=1.0)
    argp.add_argument("-ne", "--effpopsize", type=float, default=1000.0)
    argp.add_argument("-o", "--output", type=ap.FileType("w"),
                      default=sys.stdout)

    args = argp.parse_args(args)
    snps = [x.strip() for x in args.snps]
    scores = get_scores(args.sscores)

    effects = []
    for snp in snps:
        score = scores[snp] if snp in scores else  0.0
        S = 4 * args.effpopsize * score
        eps = rdm.gauss(0.0, args.epsstd)
        delta = 1.0 #rdm.choice([1.0, -1.0])
        z = delta * (S ** args.tau) * (1 + eps)
        effects.append(z)

    # we need to normalize so narrow-sense heritablity is ~ h2
    bvar = sum(e ** 2 for e in effects)
    C = args.h2 / bvar

    for idx, e in enumerate(effects):
        z = e * math.sqrt(C)
        snp = snps[idx]
        args.output.write("{} {}".format(snp, z) + os.linesep)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
