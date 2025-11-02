  main()
  _":= "__main___name__ =

if eline()ipnsive_pcompreheun_peline.r  pi  ipeline()
izationPalhensiveVisupreome = Cpelin   pi
 """nctionexecution fu"Main "" in():
   

def maraise         
   }"): {str(e)on pipelinetiizasualrror in vi"E    print(f:
        tion as except Excep    e 
              ")
  componentse dashboardctivtera"- In     print(       ions")
izatvisualattern gement pga ent("- Social      prin")
      hartsparison cty comthentici Aut("-        prin
    s:")izationated visualnGener\"print(            ")
y!essfullpleted succeline comon Pip Visualizatisive"Comprehen  print(          * 60)
 print("="    
              s()
      alizationisut_v_engagemensocialcreate_       self.s
     tiont visualizaengagemenate social   # Cre         
          arts()
   arison_chnticity_comphereate_autself.c    
        chartsn socity compariuthentireate a         # C  try:
    
            
 0) * 6rint("="     p.")
   peline..tion Pizaalihensive Visurting Compre print("Sta    "
   ine""n pipeltioizaplete visualRun the com """
       ):eline(selfensive_pipcompreh def run_  
   
  se()lt.clo p   
    ight')inches='t00, bbox_      dpi=3         , 
    ng'ysis.pnt_anal/sentimengagemential_e/socizationsig('visualef     plt.sav
   out()_lay plt.tight            
  tion=45)
 , rotais='x'params(axk_ticaxes[1].        ')
    'Valueel(labs[1].set_y         axe   atistics')
 Analysis Stententim'Sitle(xes[1].set_t     a])
       e22'#e67#f39c12', '59b6', ''#9b['#3498db', lor=coues, s, valtric[1].bar(me     axes            
      , 0)]
 ctivity_std'et('subjement_dist.g      senti                0),
_mean',tivityt('subject.gedisnt_ sentime                 ,
   ', 0)td'polarity_sist.get(sentiment_d                
     n', 0),arity_meaolet('piment_dist.g= [sent    values      ']
   ity Std 'Subjectivn', Meaectivity'Subj, d'y StPolarit Mean', 'Polarity [' metrics =          t:
 timent_dis     if sen)
   tion', {}distribument_ntiet('sement_data.gdist = sentisentiment_s
        ic statistiment   # Sent  
     )
      omments'on in Cistributiiment Dnt('Overall Se.set_title0]     axes[rs)
   olors=colo%1.1f%%', c='topcts, aumentabels=senti, l.pie(countsxes[0]     a   
   
     5a5a6']c3c', '#9 '#e74', ['#2ecc71lors = co0)]
       utral', 'net.get(enl_sentim overal           ),
     e', 0t('negativgent.timesen  overall_               ,
, 0)ositive'et('p.g_sentimentlloveraounts = [      c']
  ral 'Neutgative',ve', 'NesitiPo = ['entssentim
        tionistribut dall sentimen    # Over    
    ))
    (15, 6figsize=plots(1, 2, t.subaxes = pl  fig, 
      pie charttribution ntiment dis    # Se
        eturn
          r
      sentiment:t overall_     if no  
 ent', {})erall_sentima.get('ovent_datnt = sentimimerall_sent     ove 
      eturn
            r_data:
    ntot sentime       if n})
 ysis', {ment_anal.get('sential_analysisself.sociata = ntiment_d
        se""ions"alizatsis visuent analy sentimCreate""      "(self):
  alizationsment_visucreate_senti    def _  
()
      alizationsment_visute_sentilf._crea   sen
     lizatios visuaysialentiment aneate s       # Cr       
 e()
   plt.clos  
    t')inches='tighox_i=300, bb        dp         , 
  erview.png'rns_ovattet_p/engagemen_engagementtions/socialualizasavefig('visplt.
        )t(.tight_layou  plt    
      )
    t'ment CounComnt Score vs le('Engageme].set_tites[1, 1         ax
        Count')mmentylabel('Co].set_  axes[1, 1              nt Score')
l('Engagemeet_xlabe 1].s     axes[1,           )
0.6 alpha=nt_count'],commeta[' sample_daore'],'scsample_data[r(attees[1, 1].sc    ax        
    s:ta.column sample_dant_count' inmecomumns and 'ata.colmple_dre' in sascoif '            '])))
ntemeag'engocial_data[f.s, len(sel(1000e(n=minment'].samplngagedata['eelf.social_ = sample_data           s_data:
 f.socialin selgagement' ata and 'enial_d.soc  if selfip
      ionshlatt reommencore vs cEngagement s   #    
         ion')
 e DistributContent Typtitle('.set_axes[1, 0]
        olors)lors=c1f%%', coopct='%1. aut                     s], 
ype_t content) for ct in.title(e('_', ' ')t.replac[cs, labels=(countie0].p axes[1,    
           ypes]
  content_tr ct inntage'] force[ct]['peent_by_types = [engagem percentage      _types]
 in content for ct ']]['countt_by_type[ctgagemennts = [en    couon
    utirib type distntent# Co 
        
       n=45) rotatioxis='x',_params(a].tick 1es[0,   ax        )
 ents'erage Commylabel('Avet_xes[0, 1].s      ae')
      t Typnt by Conten Comment Couageveret_title('As[0, 1].saxe       )
     =colorsnts, colors, comment_type(conte].bar 10,       axes[   pes]
  n content_ty'] for ct it']['meanunmment_co[ct]['co_typegement_by= [engacomments         
    types[0]]:ontent_y_type[ct_b engagemenunt' incoif 'comment_
        ent countsComm  #   
           45)
 ion=otat, r='x'_params(axis 0].tickaxes[0,     )
   ore'e Scagabel('Aver].set_ylaxes[0, 0        Type')
 by Content ent Scoreagem'Average Engt_title(s[0, 0].se   axe   =colors)
  lorscores, cos, ntent_typecobar( axes[0, 0].s]
       ent_typen cont] for ct ian'core']['me_type[ct]['st_bygagemencores = [en
        stributionisres dgement sconga  # E
            
  #99ff99']6b3ff', 'ff9999', '#6colors = ['#   ys())
     ket_by_type.emen= list(engagpes ontent_ty c
              d')
 ht='bolfontweig, ize=16 fonts',ent Typesss Cont Acronsement Pattergag'Social Entitle(.sup       fig)
 2), 1(15, figsize=ots(2, 2ubpl plt.ses = fig, axs
       on chartomparis ce engagementat      # Cre   
  urn
     ret           _type:
 bynt_gagemef not en i         
  
    _type', {})ent_by'engagemta.get(_daagementype = engagement_by_tng
        eis', {})ysal_angement('engais.getysocial_analata = self.st_dmenage eng     
    rn
      etu       r   nalysis:
  lf.social_anot seif             
  ")
  lizations...uat visenl engagemng sociant("Creati pri"
       "lizations"ern visuaement pattengagcial eate so""Cr     "   ):
zations(selfliement_visuaengagial_f create_soc
    de  
      ard.html')_dashboenticityve/autheracti/intalizationse_html('visu    fig.writ          
  
ashboard")s D Analysity Authenticiactiventerle_text="I       tit                False,
  nd=0, showlegeight=80te_layout(he   fig.upda    
         )
ow=2, col=24ecdc4'), rer_color='#rk          ma                    
 al',ts, name='Real_counrent_types, y=ar(x=contece(go.Bg.add_tra  fi       
   2, col=2)w=ff6b6b'), rolor='#coarker_        m                  
     Fake',s, name='_counts, y=faketent_type=conBar(xd_trace(go..ad         fig  
          _types]
   n contentfor ct i) osts', 0t('real_pgeodal[ct].cross_m = [unts   real_co       t_types]
  ct in contenr osts', 0) fo.get('fake_pmodal[ct]s = [cross_  fake_count        ))
  modal.keys(ross_t(c_types = lisnt  conte       dal:
   ss_moro c
        ifernsttoss-modal pa      # Cr        
  1)
, col=ow=2dc4']), rb', '#4ec#ff6b6olor=['er_c  mark                      nts',
   vg Commes, name='Ag_comment, y=avr(x=labelsrace(go.Ba fig.add_t
               s', 0)]
_comment'avg{}).get(el.get('1', abt_by_lagemenng  e                     , 0),
ents'comm'avg_et(t('0', {}).gabel.geent_by_l [engagemomments =  avg_c
      ments   # Com   
         1, col=2)
  row=ecdc4']),ff6b6b', '#4or=['#er_colark  m                        e',
  Scors, name='Avgcore=avg_sbels, y.Bar(x=lad_trace(go  fig.ad    
      )]
    e', 0scor'avg_, {}).get(et('1'_by_label.gentagemeng                   
  , 0),'avg_score'', {}).get(t('0.geent_by_label= [engagemes _scorvg      as
  orengagement sc E      # 
  
       1, col=1), row=']), '#4ecdc4r=['#ff6b6b'arker_colo         m                 
  e='Posts',counts, namy=abels, (x=lace(go.Barig.add_tr
        f        ]
0)t', }).get('coun1', {label.get('agement_by_      eng     , 
      'count', 0)', {}).get(t('0l.geent_by_labengagems = [entcou       , 'Real']
 ake'bels = ['F   lan
     tributioisntent d   # Co
     
                )]
ar"}]": "b{"type: "bar"}, {"type"    [             r"}],
  type": "ba"bar"}, {""type": [[{specs=         s'),
   Patternoss-Modal vity', 'Cr Actiomment'C                     es', 
     oragement Scution', 'Eng Distrib'Contentes=(_titl    subplot,
        2, cols=2   rows=        ubplots(
 g = make_s     fiigure
   e subplot f     # Creat""
   n chart"comparisoy itictive authente interac"""Creat       al):
 model, cross_labent_by_elf, engagemrt(schathenticity_e_auractiv_inte_createdef 
    
        ross_modal), c_by_labelgagementty_chart(enenticiractive_auth_create_inteelf.n
        sisoomparticity ctive authenracCreate inte#            
  
   close()     plt.
   ht')nches='tigx_ipi=300, bbo          d         
 .png',overviewarison_compcity_tilysis/authenty_ananticiations/authe'visualizsavefig(        plt.t()
tight_layou plt.    
          d()
 legen]. 1    axes[1,    =45)
    rotationent_types], t in contle() for c' ').titce('_', s([ct.replaxticklabel, 1].set_ axes[1   
        (x)et_xticks1].ses[1,   ax       
   osts')er of Pmbbel('Nu, 1].set_yla    axes[1       
 ty')nticiy Authetion bDistribue tent Typt_title('Con, 1].se axes[1
           ecdc4')or='#4al', col label='Re, width,real_counts, x + width/2 1].bar(s[1,xe           a
 ='#ff6b6b')e', color, label='Fakthts, wid_coun/2, fakear(x - width, 1].bes[1ax              
    
       = 0.35   width         )
t_types)en(contrange(len   x = np.a         
          _types]
  t in content) for csts', 0'real_poct].get(cross_modal[ts = [real_coun            ent_types]
t in conts', 0) for cost_pfake('getal[ct].ross_modcounts = [c   fake_      eys())
   s_modal.kt(crosypes = lis_t    content    
    ss_modal:if cro        s', {})
ternss_modal_patet('croauth_data.gl = oss_moda   cr
     henticity by auttionibu distrtent typeon C
        #      
  old')eight='b fontwnter','ce', ha='{v:.1f})*0.01, ftsencomm(avg_v + maxext(i, xes[1, 0].t           a
 ments):e(avg_com in enumerator i, v    f
    Comments')rage abel('Ave, 0].set_yl    axes[1   icity')
 uthentts by ACommenrage tle('Ave.set_ti[1, 0]      axesc4'])
  ecdb', '#4lor=['#ff6b6ments, co_com avglabels,, 0].bar(s[1   axeents
     erage comm  # Av   
      ld')
     eight='boontwcenter', f, ha='f'{v:.1f}', 0.01)*x(avg_scores(i, v + matextes[0, 1].      axes):
      avg_scorrate(numer i, v in e    fo   
 ')reSco('Average .set_ylabeles[0, 1]        axty')
y Authenticiment Score bgagerage Ene('Ave.set_titl[0, 1]     axesdc4'])
   ecb6b', '#4f6=['#fs, colorg_scorels, avar(labe1].b  axes[0,       
resscoement ge engageraAv #       
 
        t='bold')eighter', fontw'cenha=', 01, f'{v:,}nts)*0.v + max(cou 0].text(i, s[0,   axe         ):
ntsate(counumeri, v in e        for s')
 Postr ofbel('Numbeet_yla0, 0].s axes[     )
  thenticity' by Autribution'Content Disitle( 0].set_t axes[0,       '])
#4ecdc4, 'ff6b6b'or=['#nts, col(labels, cou[0, 0].barxes        aution
t distrib  # Conten   
          0)]
  ',commentsvg_'a1', {}).get(et('.glabel_by_ement       engag       
         0),ments', vg_com {}).get('ael.get('0',ment_by_labengageomments = [g_c       av', 0)]
 ('avg_score.getget('1', {})l.nt_by_labegemenga      e               ,
ore', 0)get('avg_sc}).get('0', {bel._by_lagementgas = [eng_score
        avount', 0)]et('c', {}).gel.get('1nt_by_labgagemeen                0), 
  ('count',}).getget('0', {_by_label.gements = [enga    count    eal (1)']
 'Rke (0)',abels = ['Fa       lg
 plottin data for  # Extract           
bold')
    ht='ntweigfoze=16, ns', fontsi Patterontentke vs Real C Fais:ity Analysnticitle('Authe.supt fig
       5, 12))igsize=(1, fs(2, 2.subplot= pltes    fig, axharts
     arison cmpticity coate authen   # Cre    
     
    eturn  r
          vailable")ata ay label dnt bmeo engage   print("N         l:
_by_labementage not eng     if     
   
   bel', {})nt_by_lat('engagemea.gel = auth_dat_by_labeagement       eng, {})
 lysis'icity_anants.get('authelysial_anaf.socia = selatuth_d a       y data
henticitutct a  # Extra      
        
rn     retu       ailable")
ysis data avcial anal"No so   print(    :
     l_analysisialf.socf not se      i        
  .")
n charts..parisocomty nticig authent("Creatinri p
       terns"""ontent pat real c fake vsowings shrison chart compaityicrate authent""Gene    "lf):
    _charts(searison_compityte_authenticef crea
    durn {}
       ret         ")
ts not foundcial datasening: Sorint("War         pdError:
   FileNotFounxcept 
        edatasetsturn   re       uet")
   iment.parqntents_with_semmagement/co/social_engata"processed_duet(rqad_pa.re = pds']comment['datasets        ")
    parquett_data.gemenegrated_engaintent/ngagemta/social_eed_dat("processread_parquement'] = pd.ets['engage       datas    ry:
  t
       ts = {}    datase""
    "tsent dataseial engagem"Load soc"
        "):tasets(selfal_daad_soci _lo   def
    
 return {}            und")
not fodatasets  Clean ("Warning: print          undError:
 tFoxcept FileNo
        e datasets   return  ")
       n.parqueteanal_clidation_fitasets/vallean_da/cessed_data"proct(d_parque] = pd.readation'sets['valiata         d
   parquet")al_clean.in_fasets/testan_datata/cleed_docesst("prarquepd.read_p= '] ets['testatas        d   quet")
 al_clean.parts/train_finlean_dataseta/cessed_dat("proc_parquead pd.re'train'] =sets[data           try:
        {}
  atasets =      d  "
""atasets clean doad""L        "f):
ts(selatasean_dad_clef _lo    
    dern {}
      retu    
  nd")not fouilepath} : {ff"Warning    print(      r:
  rroFoundENotept File        exc(f)
son.load  return j             r') as f:
 epath, '(filth open          wi:
       trys"""
   s resultON analysid JSLoa""":
        epath)n(self, fil _load_jso  def       

   ts()tase_daoad_social= self._locial_data .sself        datasets()
_clean_oadta = self._lclean_da     self.
   setsocessed data Load pr
        #)
        json"sis.ping_analy/id_mapmage_catalogs/isultlysis_re_json("ana._loadysis = self.image_analself        ")
s.jsonion_analysiegratnton/text_itegratilts/text_innalysis_resu"an(jsoself._load_sis = .text_analy      self  .json")
ist_analys_engagemenis/socialalysanults/social_ysis_resjson("anaload_self._l= ysis al_analoci  self.sts
      sul reanalysisoad     # L    
        zations"
isualiath / "v self.base_pons_path =isualizati.v      self_data"
   "processedase_path / = self.bed_data_pathself.process    
    s_results"ysi/ "analf.base_path el sts_path =resulsis_aly  self.an      (".")
= Pathase_path .b    self:
    lf)_(sef __init_ine:
    detionPipelVisualizaprehensiveclass Comusl")

palette("h
sns.set_n-v0_8')se('seabor.ut.styletlib
plmatployle for 
# Set stgnore')
('irwarningsings.filte
warnt warningsPath
imporib import 
from pathl osport json
im
importlots make_subprtbplots impootly.su plrom as go
fcts_objegraphort plotly.
imps as px.expresport plotlys
imaborn as snrt seas plt
impo.pyplot otlibatplimport mumpy as np
ort nimp as pd
mport pandas""

ius
"enticity focth with ausis resultsd analyeteompll ctions for alzali visuaeates
Cral Analysisor Multimodne fion PipeliVisualizatve nsi"
Comprehehon3
""v pyt!/u