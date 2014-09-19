#! /usr/bin/env python
import argparse as ap
import math
import os
import random as rdm
import sys

import numpy as np

MAJOR = 0
MINOR = 2
HETER = 1


def gen_fam(num_pop, prefix):
    fam = []
    with open("{}.fam".format(prefix), "w") as famfile:
        counter = 0
        for idx in range(num_pop):
            fid = "FID{}".format(idx)
            iid = "IID{}".format(idx)
            wf = str(counter)
            wm = str(counter + 1)
            sex = rdm.choice(['1', '2']) # this might be better as 0
            pheno = "0" # dummy value.
            row = [fid, iid, wf, wm, sex, pheno]
            famfile.write(" ".join(row) + os.linesep)
            fam.append(row)
            counter += 2

    return fam


def gen_map(num_vars, prefix):
    chcode = "0"
    pmorgans = "0"
    alleles = ["A", "C", "G", "T"]
    map = []
    with open("{}.map".format(prefix), "w") as mapfile:
        for id in range(num_vars):
            vid = "SNP{}".format(id)
            minor, major = rdm.sample(alleles, 2)
            row = [chcode, vid, pmorgans, str(id), minor, major] 
            line = " ".join(row)
            map.append(row)
            mapfile.write(line[:-3] + os.linesep)

    return map

def gen_ped(g, fam, map, prefix):
    n, v = g.shape
    with open("{}.ped".format(prefix), "w") as pedfile:
        for idx, row in enumerate(g):
            famstr = fam[idx]
            calls = []
            for jdx, rcall in enumerate(row):
                minor, major = map[jdx][-2], map[jdx][-1]
                if rcall == MAJOR:
                    output = "{} {}".format(major, major)
                elif rcall == HETER:
                    output = "{} {}".format(major, minor)
                elif rcall == MINOR:
                    output = "{} {}".format(minor, minor)
                else:
                    sys.stderr.write("INVALID CALL!")
                    return 1
                calls.append(output)
        
            final = "{} {}".format(" ".join(famstr), " ".join(calls))
            pedfile.write(final + os.linesep)

    return


def main(args):
    argp = ap.ArgumentParser(description="Simulate a random genotype")
    argp.add_argument("n", type=int, help="Number of samples.")
    argp.add_argument("m", type=int, help="Number of SNPs.")
    argp.add_argument("f", type=float, help="MAF") # fine for now; but should change to accept a list/file at some point
    argp.add_argument("prefix", help="Output prefix for the ped map fam")
    # not used atm
    #argp.add_argument("-o", "--output", type=ap.FileType("w"),
    #                 default=sys.stdout)

    args = argp.parse_args(args)

    # right now we have fixed minor allele frequency.
    # it would be nice to update to accept a list of alpha/beta shape
    # parameters and sample from a beta
    g = np.random.binomial(2, args.f, size=(args.n, args.m))
    n, m = g.shape

    fam = gen_fam(n, args.prefix)
    map = gen_map(m, args.prefix)
    gen_ped(g, fam, map, args.prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
