import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 页面设置
# ==========================================

st.set_page_config(
    page_title="胎儿出生体重预测系统",
    page_icon="👶",
    layout="wide"
)

st.title("👶 Birth Weight Predictive System")
st.markdown("Based on XGBoost +  Calibration Model")

# ==========================================
# 加载模型
# ==========================================

xgb_model = joblib.load(
    "XGBoost_36-39week_model.joblib"
)

calibration_model = joblib.load(
    "XGBoost_36-39week_calibration_model.joblib"
)

# ==========================================
# 特征定义
# ==========================================

FEATURES = {

    "Maternal Information": {

        "Maternal age (years)": "del_mother_age",

        "Maternal anemia(0=none,1=yes)": "del_mother_anemia_g",

        "Maternal diabetes(0=none,1=yes)": "del_mother_dia",

        "Gravidity(times)": "del_mother_gravidity_g",

        "Hypertensive disorders of pregnancy(0=none,1=yes)":
            "del_mother_hpd_g1",

        "Parity(times)":
            "del_mother_parity_g",

        "Maternal height (cm)":
            "file_mother_height",

        "Pre-pregnancy weight (kg)":
            "file_mother_prepreg_weight",

        "Conception method":
            "file_mother_conception",

        "Maternal education level(1:middle school or below,2:high school,3:college degree or above)":
            "file_mother_edu_g"
    },

    "Paternal Information": {

        "Paternal height (cm)":
            "father_height",

        "Paternal weight (kg)":
            "father_weight"
    },

    "Precheck Information": {

        "Fundal height at 36 weeks (cm)":
            "precheck_mother_fh_36",

        "Fundal height at 38 weeks (cm)":
            "precheck_mother_fh_38",

        "Weight gain at 36 weeks (kg)":
            "precheck_mother_weight_change_36",

        "Weight gain at 39 weeks (kg)":
            "precheck_mother_weight_change_39",

        "Maternal weight at 36 weeks (kg)":
            "precheck_mother_weight_36",

        "Maternal weight at 38 weeks (kg)":
            "precheck_mother_weight_38"
    },

    "Amniotic Fluid Information": {

        "AFI at 36 weeks":
            "uls_afi_36",

        "AFI at 39 weeks":
            "uls_afi_39",

        "MVP at 37 weeks":
            "uls_mvp_37",

        "MVP at 38 weeks":
            "uls_mvp_38",

        "MVP at 39 weeks":
            "uls_mvp_39"
    },

    "Ultrasound Information": {

        "AC at 36 weeks(cm)":
            "uls_fetal_ac_36",

        "AC at 37 weeks(cm)":
            "uls_fetal_ac_37",

        "AC at 38 weeks(cm)":
            "uls_fetal_ac_38",

        "AC at 39 weeks(cm)":
            "uls_fetal_ac_39",

        "BPD at 36 weeks(cm)":
            "uls_fetal_bpd_36",

        "BPD at 38 weeks(cm)":
            "uls_fetal_bpd_38",

        "FL at 36 weeks(cm)":
            "uls_fetal_fl_36",

        "FL at 37 weeks(cm)":
            "uls_fetal_fl_37",

        "FL at 38 weeks(cm)":
            "uls_fetal_fl_38",

        "HC at 36 weeks(cm)":
            "uls_fetal_headcir_36",

        "HC at 37 weeks(cm)":
            "uls_fetal_headcir_37",

        "HC at 38 weeks(cm)":
            "uls_fetal_headcir_38",

        "HC at 39 weeks(cm)":
            "uls_fetal_headcir_39",

        "Placental thickness at 36 weeks(cm)":
            "uls_placental_thickness_36",

        "Placental thickness at 38 weeks(cm)":
            "uls_placental_thickness_38",

        "Placental thickness at 39 weeks(cm)":
            "uls_placental_thickness_39"
    }
}

# ==========================================
# 输入区域
# ==========================================

input_data = {}

for section_name, section_features in FEATURES.items():

    st.header(section_name)

    cols = st.columns(3)

    i = 0

    for display_name, model_name in section_features.items():

        with cols[i % 3]:

            # 二分类变量

            if model_name in [
                "del_mother_anemia_g",
                "del_mother_dia",
                "del_mother_hpd_g1"
            ]:

                value = st.selectbox(
                    display_name,
                    [0, 1],
                    key=model_name
                )

            # 受孕方式

            elif model_name == "file_mother_conception":

                option = st.selectbox(
                    display_name,
                    [
                        "Natural conception",
                        "ART"
                    ],
                    key=model_name
                )

                value = 0 if option == "Natural conception" else 1

            else:

                value = st.number_input(
                    display_name,
                    value=0.0,
                    key=model_name
                )

            input_data[model_name] = value

            i += 1

# ==========================================
# 预测按钮
# ==========================================

if st.button("Begin To Predict"):

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=xgb_model.get_booster().feature_names)#确保预测时变量顺序与训练的模型一致

    raw_pred = xgb_model.predict(
        input_df
    )

    final_pred = calibration_model.predict(
        raw_pred.reshape(-1, 1)
    )[0]

    st.divider()

    st.metric(
        "Estimated Fetal Weight",
        f"{final_pred:.0f} g"
    )

    if final_pred < 2500:

        st.error(
            "Low Birth Weight（<2500 g）"
        )

    elif final_pred > 4000:

        st.warning(
            "Macrosomia（>4000 g）"
        )

    else:

        st.success(
            "Normal birth weight"
        )
