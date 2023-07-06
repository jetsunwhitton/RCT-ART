import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import MultiComparison
from scipy import stats

def across_models_overall():
    biobert_runs = pd.read_csv("../evaluation_results/models_all_domains/biobert/runs.csv")
    bb_fs = biobert_runs[["NER_F","REL_F","JOINT_F","STRICT_F","RELAXED_F"]]

    scibert_runs = pd.read_csv("../evaluation_results/models_all_domains/scibert/runs.csv")
    sb_fs = scibert_runs[["NER_F","REL_F","JOINT_F","STRICT_F","RELAXED_F"]]

    roberta_runs = pd.read_csv("../evaluation_results/models_all_domains/roberta/runs.csv")
    rb_fs = roberta_runs[["NER_F","REL_F","JOINT_F","STRICT_F","RELAXED_F"]]

    ner = pd.concat([bb_fs["NER_F"],sb_fs["NER_F"],rb_fs["NER_F"]],axis=1,keys=["biobert","scibert","roberta"])
    ner = ner.reset_index().rename(columns={'index': 'run'})
    ner_long = pd.melt(ner, id_vars='run', var_name='model', value_name='f1_score')
    print(ner)
    print(ner_long)


    rel = pd.concat([bb_fs["REL_F"],sb_fs["REL_F"],rb_fs["REL_F"]],axis=1,keys=["biobert","scibert","roberta"])
    rel = rel.reset_index().rename(columns={'index': 'run'})
    rel_long = pd.melt(rel, id_vars='run', var_name='model', value_name='f1_score')
    print(rel)
    print(rel_long)

    joint = pd.concat([bb_fs["JOINT_F"],sb_fs["JOINT_F"],rb_fs["JOINT_F"]],axis=1,keys=["biobert","scibert","roberta"])
    joint = joint.reset_index().rename(columns={'index': 'run'})
    joint_long = pd.melt(joint, id_vars='run', var_name='model', value_name='f1_score')

    strict = pd.concat([bb_fs["STRICT_F"],sb_fs["STRICT_F"],rb_fs["STRICT_F"]],axis=1,keys=["biobert","scibert","roberta"])
    strict = strict.reset_index().rename(columns={'index': 'run'})
    strict_long = pd.melt(strict, id_vars='run', var_name='model', value_name='f1_score')

    relaxed = pd.concat([bb_fs["RELAXED_F"],sb_fs["RELAXED_F"],rb_fs["RELAXED_F"]],axis=1,keys=["biobert","scibert","roberta"])
    relaxed = relaxed.reset_index().rename(columns={'index': 'run'})
    relaxed_long = pd.melt(relaxed, id_vars='run', var_name='model', value_name='f1_score')

    ner_anova = AnovaRM(ner_long, 'f1_score', 'run', within=['model'])
    ner_res = ner_anova.fit()

    rel_anova = AnovaRM(rel_long, 'f1_score', 'run', within=['model'])
    rel_res = rel_anova.fit()

    joint_anova = AnovaRM(joint_long, 'f1_score', 'run', within=['model'])
    joint_res = joint_anova.fit()

    strict_anova = AnovaRM(strict_long, 'f1_score', 'run', within=['model'])
    strict_res = strict_anova.fit()

    relaxed_anova = AnovaRM(relaxed_long, 'f1_score', 'run', within=['model'])
    relaxed_res = relaxed_anova.fit()

    print("NER_RM_ANOVA:\n",ner_res.summary())
    print("REL_RM_ANOVA:\n",rel_res.summary())
    print("JOINT_RM_ANOVA:\n",joint_res.summary())
    print("STRICT_RM_ANOVA:\n",strict_res.summary())
    print("RELAXED_RM_ANOVA:\n",relaxed_res.summary())

    with open('../statistics_outputs/anova_summaries.txt', 'w') as f:
        f.write("NER_RM_ANOVA:\n")
        f.write(str(ner_res.summary()))
        f.write("\n")

        f.write("REL_RM_ANOVA:\n")
        f.write(str(rel_res.summary()))
        f.write("\n")

        f.write("JOINT_RM_ANOVA:\n")
        f.write(str(joint_res.summary()))
        f.write("\n")

        f.write("STRICT_RM_ANOVA:\n")
        f.write(str(strict_res.summary()))
        f.write("\n")

        f.write("RELAXED_RM_ANOVA:\n")
        f.write(str(relaxed_res.summary()))

        f.close()

    # Post-hoc test using Bonferroni correction for NER
    mc_ner = MultiComparison(ner_long['f1_score'], ner_long['model'])
    mc_ner_results = mc_ner.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_ner_results[0])

    # Post-hoc test using Bonferroni correction for REL
    mc_rel = MultiComparison(rel_long['f1_score'], rel_long['model'])
    mc_rel_results = mc_rel.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_rel_results[0])

    # Post-hoc test using Bonferroni correction for JOINT
    mc_joint = MultiComparison(joint_long['f1_score'], joint_long['model'])
    mc_joint_results = mc_joint.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_joint_results[0])

    # Post-hoc test using Bonferroni correction for STRICT
    mc_strict = MultiComparison(strict_long['f1_score'], strict_long['model'])
    mc_strict_results = mc_strict.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_strict_results[0])

    # Post-hoc test using Bonferroni correction for RELAXED
    mc_relaxed = MultiComparison(relaxed_long['f1_score'], relaxed_long['model'])
    mc_relaxed_results = mc_relaxed.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_relaxed_results[0])

    with open('../statistics_outputs/posthoc_results.txt', 'w') as f:
        f.write("NER_POSTHOC:\n")
        f.write(str(mc_ner_results[0]))
        f.write("\n")

        f.write("REL_POSTHOC:\n")
        f.write(str(mc_rel_results[0]))
        f.write("\n")

        f.write("JOINT_POSTHOC:\n")
        f.write(str(mc_joint_results[0]))
        f.write("\n")

        f.write("STRICT_POSTHOC:\n")
        f.write(str(mc_strict_results[0]))
        f.write("\n")

        f.write("RELAXED_POSTHOC:\n")
        f.write(str(mc_relaxed_results[0]))

        f.close()

def across_models_ner_labels():
    biobert_runs = pd.read_csv("../evaluation_results/models_all_domains/biobert/labels/ner/runs.csv")
    bb_fs = biobert_runs[["OC_F", "INTV_F", "MEAS_F"]]

    scibert_runs = pd.read_csv("../evaluation_results/models_all_domains/scibert/labels/ner/runs.csv")
    sb_fs = scibert_runs[["OC_F", "INTV_F", "MEAS_F"]]

    roberta_runs = pd.read_csv("../evaluation_results/models_all_domains/roberta/labels/ner/runs.csv")
    rb_fs = roberta_runs[["OC_F", "INTV_F", "MEAS_F"]]

    oc = pd.concat([bb_fs["OC_F"],sb_fs["OC_F"],rb_fs["OC_F"]],axis=1,keys=["biobert","scibert","roberta"])
    oc = oc.reset_index().rename(columns={'index': 'run'})
    oc_long = pd.melt(oc, id_vars='run', var_name='model', value_name='f1_score')

    intv = pd.concat([bb_fs["INTV_F"],sb_fs["INTV_F"],rb_fs["INTV_F"]],axis=1,keys=["biobert","scibert","roberta"])
    intv = intv.reset_index().rename(columns={'index': 'run'})
    intv_long = pd.melt(intv, id_vars='run', var_name='model', value_name='f1_score')

    meas = pd.concat([bb_fs["MEAS_F"],sb_fs["MEAS_F"],rb_fs["MEAS_F"]],axis=1,keys=["biobert","scibert","roberta"])
    meas = meas.reset_index().rename(columns={'index': 'run'})
    meas_long = pd.melt(meas, id_vars='run', var_name='model', value_name='f1_score')

    print(oc_long)
    oc_anova = AnovaRM(oc_long, 'f1_score', 'run', within=['model'])
    oc_res = oc_anova.fit()

    intv_anova = AnovaRM(intv_long, 'f1_score', 'run', within=['model'])
    intv_res = intv_anova.fit()

    meas_anova = AnovaRM(meas_long, 'f1_score', 'run', within=['model'])
    meas_res = meas_anova.fit()

    print("OC_RM_ANOVA:\n",oc_res.summary())
    print("INTV_RM_ANOVA:\n",intv_res.summary())
    print("MEAS_RM_ANOVA:\n",meas_res.summary())

    with open('../statistics_outputs/across_models_ner_labels.txt', 'w') as f:
        f.write("OC_RM_ANOVA:\n")
        f.write(str(oc_res.summary()))
        f.write("\n")

        f.write("INTV_RM_ANOVA:\n")
        f.write(str(intv_res.summary()))
        f.write("\n")

        f.write("MEAS_RM_ANOVA:\n")
        f.write(str(meas_res.summary()))

        f.close()

    # Post-hoc test using Bonferroni correction for OC
    mc_oc = MultiComparison(oc_long['f1_score'], oc_long['model'])
    mc_oc_results = mc_oc.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_oc_results[0])

    # Post-hoc test using Bonferroni correction for INTV
    mc_intv = MultiComparison(intv_long['f1_score'], intv_long['model'])
    mc_intv_results = mc_intv.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_intv_results[0])

    # Post-hoc test using Bonferroni correction for MEAS
    mc_meas = MultiComparison(meas_long['f1_score'], meas_long['model'])
    mc_meas_results = mc_meas.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_meas_results[0])

    with open('../statistics_outputs/across_models_ner_labels_posthoc.txt', 'w') as f:
        f.write("OC_POSTHOC:\n")
        f.write(str(mc_oc_results[0]))
        f.write("\n")

        f.write("INTV_POSTHOC:\n")
        f.write(str(mc_intv_results[0]))
        f.write("\n")

        f.write("MEAS_POSTHOC:\n")
        f.write(str(mc_meas_results[0]))

        f.close()

def across_models_rel_labels():
    biobert_runs = pd.read_csv("../evaluation_results/models_all_domains/biobert/labels/rel/runs.csv")
    bb_fs = biobert_runs[["A1_RES_F", "A2_RES_F", "OC_RES_F"]]

    scibert_runs = pd.read_csv("../evaluation_results/models_all_domains/scibert/labels/rel/runs.csv")
    sb_fs = scibert_runs[["A1_RES_F", "A2_RES_F", "OC_RES_F"]]

    roberta_runs = pd.read_csv("../evaluation_results/models_all_domains/roberta/labels/rel/runs.csv")
    rb_fs = roberta_runs[["A1_RES_F", "A2_RES_F", "OC_RES_F"]]

    a1_res = pd.concat([bb_fs["A1_RES_F"],sb_fs["A1_RES_F"],rb_fs["A1_RES_F"]],axis=1,keys=["biobert","scibert","roberta"])
    a1_res = a1_res.reset_index().rename(columns={'index': 'run'})
    a1_res_long = pd.melt(a1_res, id_vars='run', var_name='model', value_name='f1_score')

    a2_res = pd.concat([bb_fs["A2_RES_F"],sb_fs["A2_RES_F"],rb_fs["A2_RES_F"]],axis=1,keys=["biobert","scibert","roberta"])
    a2_res = a2_res.reset_index().rename(columns={'index': 'run'})
    a2_res_long = pd.melt(a2_res, id_vars='run', var_name='model', value_name='f1_score')

    oc_res = pd.concat([bb_fs["OC_RES_F"],sb_fs["OC_RES_F"],rb_fs["OC_RES_F"]],axis=1,keys=["biobert","scibert","roberta"])
    oc_res = oc_res.reset_index().rename(columns={'index': 'run'})
    oc_res_long = pd.melt(oc_res, id_vars='run', var_name='model', value_name='f1_score')

    a1_anova = AnovaRM(a1_res_long, 'f1_score', 'run', within=['model'])
    a1_res = a1_anova.fit()

    a2_anova = AnovaRM(a2_res_long, 'f1_score', 'run', within=['model'])
    a2_res = a2_anova.fit()

    oc_anova = AnovaRM(oc_res_long, 'f1_score', 'run', within=['model'])
    oc_res = oc_anova.fit()

    print("A1_RM_ANOVA:\n",a1_res.summary())
    print("A2_RM_ANOVA:\n",a2_res.summary())
    print("OC_RM_ANOVA:\n",oc_res.summary())

    with open('../statistics_outputs/across_models_rel_labels.txt', 'w') as f:
        f.write("A1_RM_ANOVA:\n")
        f.write(str(a1_res.summary()))
        f.write("\n")

        f.write("A2_RM_ANOVA:\n")
        f.write(str(a2_res.summary()))
        f.write("\n")

        f.write("OC_RM_ANOVA:\n")
        f.write(str(oc_res.summary()))

        f.close()

    # Post-hoc test using Bonferroni correction for A1
    mc_a1 = MultiComparison(a1_res_long['f1_score'], a1_res_long['model'])
    mc_a1_results = mc_a1.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_a1_results[0])

    # Post-hoc test using Bonferroni correction for A2
    mc_a2 = MultiComparison(a2_res_long['f1_score'], a2_res_long['model'])
    mc_a2_results = mc_a2.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_a2_results[0])

    # Post-hoc test using Bonferroni correction for OC
    mc_oc = MultiComparison(oc_res_long['f1_score'], oc_res_long['model'])
    mc_oc_results = mc_oc.allpairtest(stats.ttest_rel, method='bonf')
    print(mc_oc_results[0])

    with open('../statistics_outputs/across_models_rel_labels_posthoc.txt', 'w') as f:
        f.write("A1_POSTHOC:\n")
        f.write(str(mc_a1_results[0]))
        f.write("\n")

        f.write("A2_POSTHOC:\n")
        f.write(str(mc_a2_results[0]))
        f.write("\n")

        f.write("OC_POSTHOC:\n")
        f.write(str(mc_oc_results[0]))

        f.close()

if __name__ == "__main__":
    #across_models_overall()
    across_models_ner_labels()
    across_models_rel_labels()

