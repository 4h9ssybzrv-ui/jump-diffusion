"""pension_db.py"""
SCHEME_PRESETS = {
    "nhs_2015": {"accrual_rate":1/54,"revaluation_rate":"CPI_plus_1.5","revaluation_cap":None,"normal_pension_age":67,"salary_threshold":None},
    "uss":      {"accrual_rate":1/75,"revaluation_rate":"CPI","revaluation_cap":0.10,"normal_pension_age":66,"salary_threshold":71484.0},
    "generic":  {"accrual_rate":None,"revaluation_rate":"CPI","revaluation_cap":None,"normal_pension_age":None,"salary_threshold":None},
}
def _uss_cap(cpi):
    if cpi<=0.05: return cpi
    return min(0.05+0.5*(cpi-0.05),0.10)
def _real_rr(revl,cap,cpi=0.025):
    if revl=="CPI_plus_1.5": return 0.015
    if revl=="CPI":
        if cap is not None: return (1+_uss_cap(cpi))/(1+cpi)-1
        return 0.0
    return 0.0
def calculate_db_income_projected(cfg): return float(cfg.get("projected_annual_income",0))/12
def calculate_db_income_care(cfg,long_cpi=0.025):
    pname=cfg.get("scheme_preset","generic"); p=SCHEME_PRESETS.get(pname,SCHEME_PRESETS["generic"]).copy()
    ar=float(cfg.get("accrual_rate") or p.get("accrual_rate") or 1/60)
    rr=cfg.get("revaluation_rate",p.get("revaluation_rate","CPI"))
    rc=cfg.get("revaluation_cap",p.get("revaluation_cap"))
    st=cfg.get("salary_threshold",p.get("salary_threshold"))
    if st is not None: st=float(st)
    sal=float(cfg.get("current_pensionable_salary",0))
    yn=int(cfg.get("years_of_service_to_date",0)); yr=int(cfg.get("years_of_service_at_retirement",0))
    sg=cfg.get("annual_salary_growth","CPI"); rsg=0.0 if sg=="CPI" else float(sg)
    rrr=_real_rr(rr,rc,long_cpi); total=0.0
    for y in range(1,yr+1):
        sy=sal if y<=yn else sal*((1+rsg)**(y-yn))
        if st is not None: sy=min(sy,st)
        total+=ar*sy*(1+rrr)**(yr-y)
    return total/12
def build_db_income_schedule(config):
    sched=[]
    for p in config.get("pensions",[]):
        if p.get("type") not in ("DB","hybrid"): continue
        pid=p.get("id","db"); db=p.get("db",{})
        preset=SCHEME_PRESETS.get(db.get("scheme_preset","generic"),SCHEME_PRESETS["generic"])
        npa=float(db.get("normal_pension_age") or preset.get("normal_pension_age") or 67)
        mode=db.get("input_mode","projected")
        inc=calculate_db_income_projected(db) if mode=="projected" else calculate_db_income_care(db)
        sched.append({"pension_id":pid,"normal_pension_age":npa,"monthly_income_gross":inc})
        print(f"  DB '{pid}': £{inc*12:,.0f}/yr from age {npa:.0f}  [{mode}]")
    return sched
