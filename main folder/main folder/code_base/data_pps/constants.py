import pandas as pd


class constants:
    def __init__(self):
        pass
    def __call__(self):
        FEATURES = [
            "dri_score",
            "psych_disturb",
            "cyto_score",
            "diabetes",
            "hla_match_c_high",
            "hla_high_res_8",
            "tbi_status",
            "arrhythmia",
            "hla_low_res_6",
            "graft_type",
            "vent_hist",
            "renal_issue",
            "pulm_severe",
            "prim_disease_hct",
            "hla_high_res_6",
            "cmv_status",
            "hla_high_res_10",
            "hla_match_dqb1_high",
            "tce_imm_match",
            "hla_nmdp_6",
            "hla_match_c_low",
            "rituximab",
            "hla_match_drb1_low",
            "hla_match_dqb1_low",
            "prod_type",
            "cyto_score_detail",
            "conditioning_intensity",
            "ethnicity",
            "year_hct",
            "obesity",
            "mrd_hct",
            "in_vivo_tcd",
            "tce_match",
            "hla_match_a_high",
            "hepatic_severe",
            "donor_age",
            "prior_tumor",
            "hla_match_b_low",
            "peptic_ulcer",
            "age_at_hct",
            "hla_match_a_low",
            "gvhd_proph",
            "rheum_issue",
            "sex_match",
            "hla_match_b_high",
            "race_group",
            "comorbidity_score",
            "karnofsky_score",
            "hepatic_mild",
            "tce_div_match",
            "donor_related",
            "melphalan_dose",
            "hla_low_res_8",
            "cardiac",
            "hla_match_drb1_high",
            "pulm_moderate",
            "hla_low_res_10",
        ]
        CATS = [
            "dri_score",
            "psych_disturb",
            "cyto_score",
            "diabetes",
            "tbi_status",
            "arrhythmia",
            "graft_type",
            "vent_hist",
            "renal_issue",
            "pulm_severe",
            "prim_disease_hct",
            "cmv_status",
            "tce_imm_match",
            "rituximab",
            "prod_type",
            "cyto_score_detail",
            "conditioning_intensity",
            "ethnicity",
            "obesity",
            "mrd_hct",
            "in_vivo_tcd",
            "tce_match",
            "hepatic_severe",
            "prior_tumor",
            "peptic_ulcer",
            "gvhd_proph",
            "rheum_issue",
            "sex_match",
            "race_group",
            "hepatic_mild",
            "tce_div_match",
            "donor_related",
            "melphalan_dose",
            "cardiac",
            "pulm_moderate",
        ]
        return FEATURES, CATS
