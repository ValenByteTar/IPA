const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// ── Mini renderizador Markdown (local-first, XSS-safe) ──────────────────
// Escape-first: todo pasa por esc() antes de las transformaciones, así el
// output del modelo nunca inyecta HTML. Soporta: code blocks, inline code,
// headers, bold, italic, listas, blockquotes, hr, links y math ($ / $$).
function renderMarkdown(src){
  if(!src)return '';
  const codeBlocks=[];
  // 1. Extraer code blocks ```...``` antes de escapar (preservar contenido crudo)
  let text=String(src??'').replace(/```(\w*)\n?([\s\S]*?)```/g,(m,lang,code)=>{
    codeBlocks.push({lang:lang.trim(),code});
    return `\u0000CODE${codeBlocks.length-1}\u0000`;
  });
  // 2. Escape HTML del resto
  text=esc(text);
  // 3. Math: $$...$$ (display) y $...$ (inline) → spans estilizados
  const mathSpans=[];
  text=text.replace(/\$\$([\s\S]+?)\$\$/g,(m,math)=>{
    mathSpans.push(math.trim());
    return `\u0000MATH${mathSpans.length-1}\u0000`;
  });
  text=text.replace(/\$([^\$\n]+?)\$/g,(m,math)=>{
    mathSpans.push(math);
    return `\u0000MATH${mathSpans.length-1}\u0000`;
  });
  // 4. Inline code
  text=text.replace(/`([^`\n]+)`/g,'<code class="md-code">$1</code>');
  // 5. Bold / italic
  text=text.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  text=text.replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>');
  // 6. Links [text](url)
  text=text.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  // 7. Líneas: headers, listas, quotes, hr — línea por línea
  // (el texto ya está escapado globalmente; NO re-escapar acá o destruiría
  //  los tags insertados en los pasos 4-6)
  const lines=text.split('\n');
  let html='',inUl=false,inOl=false;
  const closeLists=()=>{if(inUl){html+='</ul>';inUl=false}if(inOl){html+='</ol>';inOl=false}};
  for(const line of lines){
    const trimmed=line.trim();
    const h=trimmed.match(/^(#{1,4})\s+(.*)$/);
    const ul=trimmed.match(/^[-*]\s+(.*)$/);
    const ol=trimmed.match(/^\d+[.)]\s+(.*)$/);
    const bq=trimmed.match(/^&gt;\s?(.*)$/);
    if(/^(-{3,}|\*{3,})$/.test(trimmed)){closeLists();html+='<hr class="md-hr">';continue}
    if(h){closeLists();const l=h[1].length;html+=`<h${l+2} class="md-h">${h[2]}</h${l+2}>`;continue}
    if(ul){if(!inUl){closeLists();html+='<ul class="md-list">';inUl=true}html+=`<li>${ul[1]}</li>`;continue}
    if(ol){if(!inOl){closeLists();html+='<ol class="md-list">';inOl=true}html+=`<li>${ol[1]}</li>`;continue}
    if(trimmed.startsWith('&gt;')){closeLists();html+=`<blockquote class="md-quote">${trimmed.replace(/^&gt;\s?/,'')}</blockquote>`;continue}
    closeLists();
    if(trimmed==='')continue;
    html+=`<p class="md-p">${trimmed}</p>`;
  }
  closeLists();
  // 8. Restaurar math y code blocks
  html=html.replace(/\u0000MATH(\d+)\u0000/g,(m,i)=>`<span class="md-math">${mathSpans[Number(i)]}</span>`);
  html=html.replace(/\u0000CODE(\d+)\u0000/g,(m,i)=>{const b=codeBlocks[Number(i)];return `<pre class="md-pre${b.lang?` lang-${esc(b.lang)}`:''}"><code>${esc(b.code)}</code></pre>`});
  return html;
}

let state={};
let viewer=$('#viewer');
let selectedReport=null;
let _periodMode='days'; // 'days' | 'range' — persists across panel rebuilds
let selectedReportPath=null;
let connectionState='connecting'; // 'connected' | 'connecting' | 'disconnected'
let retryCount=0;
let retryTimer=null;
const processLabels={'scraper':'Scraper','pipeline':'Fast Path','lancedb':'LanceDB','rechunk':'Pipeline completo'};

function toast(msg){const el=$('#toast');el.textContent=msg;el.classList.add('show');setTimeout(()=>el.classList.remove('show'),2800)}
async function api(url,opts={}){const r=await fetch(url,{headers:{'Content-Type':'application/json',...(opts.headers||{})},...opts});const data=await r.json();if(!r.ok||data.error)throw Error(data.error||`HTTP ${r.status}`);return data}
function format(n){return n==null?'—':Number(n).toLocaleString('es-AR')}
function statusClass(s){return ['done','complete','available','healthy','indexed','published','approved'].includes(String(s).toLowerCase())?'ok':['error','failed','dead','rejected'].includes(String(s).toLowerCase())?'bad':'warning'}
function fmtDate(s){if(!s)return '—';try{const d=new Date(s);return d.toLocaleString('es-AR',{dateStyle:'medium',timeStyle:'short'})}catch{return esc(String(s))}}
function fmtRelative(s){if(!s)return '';try{const d=new Date(s);const diff=(Date.now()-d.getTime())/1000;if(diff<60)return 'hace '+Math.floor(diff)+'s';if(diff<3600)return 'hace '+Math.floor(diff/60)+'min';if(diff<86400)return 'hace '+Math.floor(diff/3600)+'h';return 'hace '+Math.floor(diff/86400)+'d'}catch{return ''}}

function setConnectionState(s){
  connectionState=s;
  const el=$('#connection-status');
  if(!el)return;
  if(s==='connected'){el.textContent='Conectado';el.className='conn-status ok'}
  else if(s==='connecting'){el.textContent=retryCount>0?`Reintentando (${retryCount})…`:'Conectando…';el.className='conn-status connecting'}
  else{el.textContent='Sin conexión';el.className='conn-status bad'}
}

async function checkHealth(){
  try{
    await fetch('/api/health',{signal:AbortSignal.timeout(3000)});
    return true;
  }catch{return false}
}

async function refreshAll(){
  try{
    state=await api('/api/state');
    if(connectionState!=='connected'){retryCount=0;setConnectionState('connected')}
    // Save scroll position before re-rendering
    const savedScrollX=window.scrollX, savedScrollY=window.scrollY;
    // Save scroll positions of all scrollable containers
    const scrollContainers=[];
    document.querySelectorAll('.source-days-table,.log-view,.history-list,.exec-source-days').forEach(el=>{
      scrollContainers.push({el, top:el.scrollTop, left:el.scrollLeft});
    });
    renderState();
    await loadReviewContent();
    // Auto-select latest report while pipeline is running
    const pipelineRunning=state.pipeline?.status==='running';
    if(pipelineRunning&&!selectedReportPath){
      const latest=state.reporter?.report;
      if(latest?.path){
        await loadReport(latest.path,{scroll:false});
      }
    }else if(selectedReportPath){
      await loadReport(selectedReportPath,{scroll:false});
    }
    // Restore window scroll position AFTER all DOM updates
    window.scrollTo(savedScrollX, savedScrollY);
    // Restore container scroll positions
    scrollContainers.forEach(({el,top,left})=>{
      const restored=document.querySelector(getSelectorPath(el))||el;
      if(restored){restored.scrollTop=top;restored.scrollLeft=left}
    });
    $('#last-refresh').textContent='Actualizado '+new Date().toLocaleTimeString();
  }catch(e){
    setConnectionState('disconnected');
    retryCount++;
    scheduleRetry();
    toast('Sin conexión al servidor: '+e.message);
  }
}

function getSelectorPath(el){
  if(el.id)return '#'+el.id;
  if(el.className)return '.'+el.className.split(' ').join('.');
  return el.tagName.toLowerCase();
}

function scheduleRetry(){
  if(retryTimer)return;
  const delay=Math.min(1000*Math.pow(1.5,retryCount),15000); // exponential backoff, max 15s
  setConnectionState('connecting');
  retryTimer=setTimeout(async()=>{
    retryTimer=null;
    const healthy=await checkHealth();
    if(healthy){
      refreshAll();
    }else{
      retryCount++;
      scheduleRetry();
    }
  },delay);
}

async function restartSystem(){
  const btn=$('#restart-btn');
  if(btn){btn.disabled=true;btn.textContent='Reiniciando…'}
  setConnectionState('connecting');
  toast('Reiniciando servidor del dashboard…');
  try{
    // Try to call /api/restart (works if server is up but misbehaving)
    await fetch('/api/restart',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).catch(()=>{});
  }catch{}
  // Wait and poll for the server to come back
  retryCount=0;
  for(let i=0;i<30;i++){
    await new Promise(r=>setTimeout(r,1000));
    const healthy=await checkHealth();
    if(healthy){
      if(btn){btn.disabled=false;btn.textContent='Reiniciar sistema'}
      toast('Sistema reiniciado correctamente');
      refreshAll();
      return;
    }
    setConnectionState('connecting');
  }
  if(btn){btn.disabled=false;btn.textContent='Reiniciar sistema'}
  toast('No se pudo reiniciar automáticamente. Ejecutá: .\\start_ipa_dashboard.bat');
}

function renderState(){
  const m=state.main_ingestion||{},llm=state.llm||{};
  const sc=state.scrape_counts||{total_files:0,sites:0};
  const ac=state.archive_counts||{total_files:0};
  const tc=state.transit_counts||{total_files:0};
  const mv=state.main_vector||{};
  const si=state.staging_ingestion||{};
  const sv=state.staging_vector||{};
  const pipeline=state.pipeline||{};
  const pipelineActive=pipeline.status==='running';
  $('#hero-status').innerHTML=`<span class="${llm.ready?'ok':'warning'}">●</span> ${llm.ready?'Sistema operativo':'Revisión requerida'}`;
  // Metrics: staging corpus (where fast path indexes) for chunks, main corpus for approved docs
  const scrapedFiles=sc.total_files||0;
  const archivedFiles=ac.total_files||0;
  const transitFiles=tc.total_files||0;
  const mainDocs=m.documents||0;
  const mainChunks=m.chunks||0;
  const mainLanceChunks=mv.files||mv.chunks||0;
  const stagingDocs=si.documents||0;
  const stagingChunks=si.chunks||0;
  const lanceChunks=(sv.files??sv.chunks??0)||0;
  const llmModel=llm.model_name||'—';
  const llmProvider=llm.provider||'—';
  const gpuName=(llm.gpu||'local').replace('NVIDIA GeForce RTX 4050 Laptop GPU','NVIDIA RTX');
  // Tooltips with detailed info
  const scraperProc=state.processes?.scraper||{};
  const pipelineProc=state.processes?.pipeline||{};
  const tipScraped=`Archivos sin procesar en Landing/web (zona de paso)\nScraper: ${scraperProc.status||'—'}${scraperProc.pid?' · PID '+scraperProc.pid:''}${scraperProc.detail?'\n'+scraperProc.detail:''}`;
  const tipTransit=`Procesados en Transit/ — pendientes de confirmación para el corpus principal\nTotal: ${transitFiles} archivos`;
  const tipChunks=`Chunks en Tantivy + BM25 (corpus staging)\nFast Path: ${pipelineProc.status||'—'}${pipelineProc.pid?' · PID '+pipelineProc.pid:''}${pipelineProc.detail?'\n'+pipelineProc.detail:''}\nDocumentos: ${stagingDocs}`;
  const tipLance=`Chunks vectoriales en LanceDB (corpus staging)\nLanceDB: ${lanceChunks} vectores\nDense: BGE-M3 1024 dims + Sparse`;
  const tipGPU=`GPU: ${llm.gpu||'—'}\nCUDA: ${llm.cuda_available?'Disponible':'No disponible'}\nExtensión: ${llm.extension_present?'Encontrada':'Faltante'}`;
  const tipLLM=`Modelo: ${llmModel}\nEstado: ${llm.ready?'Listo':'No listo'}\nProvider: ${llmProvider}`;
  const tipMain=`Corpus principal (aprobado)\nDocumentos: ${mainDocs}\nChunks BM25: ${mainChunks}\nChunks LanceDB: ${mainLanceChunks}\nSolo se llena al aprobar un reporte`;
  const tipArchive=`Documentos archivados en Archive/\nTotal: ${archivedFiles} archivos`;
  // Research indicator — show when agent is doing web research
  const agentResearch=state.agent_research||{};
  const ri=$('#research-indicator');
  if(agentResearch.status==='running'){
    ri.style.display='flex';
    const q=agentResearch.query||'';
    const maxU=agentResearch.max_urls||'?';
    const started=agentResearch.started_at||'';
    ri.querySelector('#research-indicator-text').textContent=`Investigación en curso: "${q}" (${maxU} fuentes máx.) — iniciada ${started}`;
  }else if(pipelineActive){
    // Ingesta en curso — visible también desde el chat, no solo en Resumen.
    ri.style.display='flex';
    const stageLabels={'scraper':'Scraping','fast_path':'Ingestando (BM25 + LanceDB)','reporter_fast':'Reporte rápido','reporter_full':'Reporte (BGE-M3 + Qwen)','parallel':'Generando reporte (BGE-M3 + Qwen)','ingestion':'Ingesta'};
    const stage=stageLabels[pipeline.stage]||pipeline.stage||'';
    ri.querySelector('#research-indicator-text').textContent=`Ingesta en curso: ${stage} · ${pipeline.percent||0}% — ${pipeline.detail||''}`;
  }else{
    ri.style.display='none';
  }
  $('#metrics').innerHTML=[
    {l:'Pendientes en Landing',v:format(scrapedFiles),s:'sin procesar',t:tipScraped,ctx:'landing'},
    {l:'En Transit',v:format(transitFiles),s:'esperan confirmación',t:tipTransit},
    {l:'Chunks indexados',v:format(stagingChunks),s:'en Tantivy + BM25 (staging)',t:tipChunks},
    {l:'chunks vectoriales',v:format(lanceChunks),s:'en LanceDB (staging)',t:tipLance},
    {l:'GPU',v:llm.ready?'Listo':'No listo',s:gpuName,t:tipGPU},
    {l:'LLM',v:llmModel,s:llmProvider,t:tipLLM},
  ].map(x=>`<div class="metric"${x.ctx?` data-ctx="${x.ctx}"`:''} title="${esc(x.t)}"><div class="label">${x.l}</div><div class="value">${x.v}</div><div class="label">${x.s}</div></div>`).join('');
  // Separate section for Corpus principal and Documentos archivados
  $('#corpus-section').innerHTML=[
    {l:'Corpus principal',v:format(mainDocs),s:'documentos',t:tipMain},
    {l:'Chunks principales',v:format(mainChunks),s:'en BM25 (aprobado)',t:`Corpus principal BM25\nChunks: ${mainChunks}`},
    {l:'Vectores principales',v:format(mainLanceChunks),s:'en LanceDB (aprobado)',t:`Corpus principal LanceDB\nVectores: ${mainLanceChunks}`},
    {l:'Documentos archivados',v:format(archivedFiles),s:'en Archive/',t:tipArchive},
  ].map(x=>`<div class="metric" title="${esc(x.t)}"><div class="label">${x.l}</div><div class="value">${x.v}</div><div class="label">${x.s}</div></div>`).join('');
  // Pipeline progress banner on overview
  if(pipelineActive){
    const stageLabels={'scraper':'Scraping','fast_path':'Ingestando (BM25 + LanceDB)','reporter_fast':'Reporte rápido','reporter_full':'Reporte (BGE-M3 + Qwen)','parallel':'Generando reporte (BGE-M3 + Qwen)'};
    const pct=pipeline.percent||0;
    const stage=stageLabels[pipeline.stage]||pipeline.stage||'';
    const detail=pipeline.detail||'';
    const bar=`<div style="height:8px;background:#1a2030;border-radius:4px;overflow:hidden;margin:8px 0"><div style="height:100%;width:${pct}%;background:linear-gradient(90deg,#56d6b0,#78a9ff);transition:width .5s"></div></div>`;
    $('#ingestions').innerHTML=`<div class="pipeline-live"><span class="eyebrow">PIPELINE EN VIVO</span><h3>${stage} · ${pct}%</h3>${bar}<p class="muted">${esc(detail)}</p>
      <div class="pipeline-live-stats">
        <div class="kv"><span class="muted">Scrapeados</span><strong>${format(scrapedFiles)}</strong></div>
        <div class="kv"><span class="muted">Docs corpus</span><strong>${format(mainDocs)}</strong></div>
        <div class="kv"><span class="muted">Chunks</span><strong>${format(stagingChunks)}</strong></div>
        <div class="kv"><span class="muted">LanceDB</span><strong>${format(lanceChunks)}</strong></div>
      </div></div>`+ingestion('Principal',m,state.main_vector);
  }else{
    $('#ingestions').innerHTML=ingestion('Principal',m,state.main_vector);
  }
  $('#llm-pill').textContent=llm.ready?'LISTO':'NO DISPONIBLE';
  $('#llm-pill').className='pill '+(llm.ready?'ok':'bad');
  $('#llm-info').innerHTML=`<p class="muted">Modelo: ${esc(llmModel)}</p><p>Provider: <strong>${esc(llmProvider)}</strong></p><p>CUDA: <strong class="${llm.cuda_available?'ok':'warning'}">${llm.cuda_available?'Disponible':'No disponible'}</strong></p><p class="muted">${esc(llm.gpu||'Sin GPU detectada')}</p>`;
  const processActions={'scraper':'/api/scraper/run','pipeline':'/api/fastpath/run','lancedb':'/api/lancedb/run','rechunk':'/api/pipeline/run'};
  // Only show the 4 canonical processes — skip web_* duplicates
  const processOrder=['scraper','pipeline','lancedb','rechunk'];
  const processDescriptions={
    scraper:'Trae contenido desde las fuentes configuradas',
    pipeline:'Parsea, chunkifica e ingesta en FastPath',
    lancedb:'Construye embeddings e índice vectorial',
    rechunk:'Ejecuta el pipeline completo de punta a punta'
  };
  const processIcons={scraper:'↓',pipeline:'⚡',lancedb:'◈',rechunk:'↻'};
  $('#processes').innerHTML=processOrder.map(k=>{
    const v=(state.processes||{})[k]||{status:'not_started'};
    const pct=v.percent!=null?` · ${v.percent}%`:'';const stage=v.stage?` · ${esc(v.stage)}`:'';
    const isRunning=v.status==='running';
    const label=processLabels[k]||k;
    const action=processActions[k];
    const statusLabel=isRunning?'Ejecutando':v.status==='done'?'Completado':v.status==='failed'?'Falló':'Listo';
    const btn=action&&!isRunning?`<button class="process-action" onclick="event.stopPropagation();startProcess('${esc(k)}','${action}')"><span>Ejecutar</span><span class="process-action-icon">▶</span></button>`:'';
    return `<div class="process-card" onclick="showProcess('${esc(k)}')"><div class="process-icon">${processIcons[k]}</div><div class="process-card-main"><div class="process-card-title"><strong>${esc(label)}</strong><span class="process-status ${statusClass(v.status)}">${statusLabel}${stage}${pct}</span></div><span class="process-description">${processDescriptions[k]}</span>${v.pid?`<span class="process-meta">PID ${v.pid}</span>`:''}</div>${btn}</div>`;
  }).join('')||'<span class="muted">No hay procesos registrados.</span>';
  renderSources();
  renderExecution();
  renderCategories(selectedReport||state.reporter?.report);
  renderHistory();
  renderTopicReview();
}

function ingestion(name,data,extra){
  return `<div class="status-row"><div><strong>${name}</strong><span class="muted"> ${data.exists?'base disponible':'sin base'}</span></div><div class="${data.exists?'ok':'warning'}">${format(data.documents)} docs · ${format(data.chunks)} chunks</div></div>`;
}

function renderSources(){
  const s=state.sources||{base:[],added:[],disabled:[]};
  const disabled=new Set(s.disabled);
  const rows=[...s.base.map(x=>({...x,active:!disabled.has(x.url),base:true})),...s.added.map(x=>({...x,active:!disabled.has(x.url)})),...s.disabled.filter(url=>!s.base.some(x=>x.url===url)&&!s.added.some(x=>x.url===url)).map(url=>({url,active:false}))];
  $('#sources').innerHTML=rows.map(x=>`<div class="source-row"><div><div class="source-url">${esc(x.url)}</div><span class="muted">${x.active?'Activa':'Desactivada'}</span></div><button class="ghost small" onclick="toggleSource('${encodeURIComponent(x.url)}',${x.active},${!!x.base})">${x.active?'Desactivar':'Activar'}</button></div>`).join('')||'<div class="notice">No hay overrides. Se utilizan las fuentes base del config.</div>';
}

function renderExecution(){
  const target=$('#execution-panel');
  if(!target)return;
  const effective=state.sources?.effective||[];
  const scraperJob=state.processes?.scraper||{};
  const reporterJob=state.processes?.web_reporter||{};
  const scraperRunning=scraperJob.status==='running';
  const reporterRunning=reporterJob.status==='running';
  const activeCount=effective.filter(s=>s.active).length;
  const avgDays=effective.length?Math.round(effective.reduce((sum,s)=>sum+(s.days_back||0),0)/effective.length):0;
  // Live counts for pipeline button status
  const sc=state.scrape||{};
  const si=state.staging||{};
  const scrapedFiles=sc.total_files||0;
  const stagingChunks=si.chunks||0;

  // If panel already has content, only update dynamic parts (status badges, buttons)
  // Don't rebuild inputs — that would reset user edits and scroll
  if(target.querySelector('.exec-period-selector')){
    // Save scroll position of the source table
    const sourceTable=target.querySelector('.exec-source-days');
    const tableScrollTop=sourceTable?sourceTable.scrollTop:0;
    // Update scraper status badge
    const scraperBadge=target.querySelector('.scraper-status');
    if(scraperBadge){
      scraperBadge.className='scraper-status '+statusClass(scraperJob.status);
      const pct=scraperJob.percent!=null?` · ${scraperJob.percent}%`:'';
      const stage=scraperJob.stage?` · ${esc(scraperJob.stage)}`:'';
      scraperBadge.textContent=scraperRunning?`ejecutando${stage}${pct}`:(scraperJob.status==='failed'||scraperJob.status==='error'?'falló':scraperJob.status==='done'?'completado':'inactivo');
    }
    // Update reporter status badge
    const reporterBadge=target.querySelector('.reporter-status');
    if(reporterBadge){
      reporterBadge.className='reporter-status '+statusClass(reporterJob.status);
      const pct=reporterJob.percent!=null?` · ${reporterJob.percent}%`:'';
      const stage=reporterJob.stage?` · ${esc(reporterJob.stage)}`:'';
      reporterBadge.textContent=reporterRunning?`ejecutando${stage}${pct}`:(reporterJob.status==='failed'||reporterJob.status==='error'?'falló':reporterJob.status==='done'?'completado':'inactivo');
    }
    // Update button disabled states
    const scraperBtn=target.querySelector('.scraper-run-btn');
    if(scraperBtn){scraperBtn.disabled=scraperRunning;scraperBtn.textContent=scraperRunning?'Scraper ejecutando…':'Ejecutar scraper'}
    const reporterBtns=target.querySelectorAll('.reporter-run-btn');
    reporterBtns.forEach(btn=>{btn.disabled=reporterRunning});
    // Update pipeline button
    const pipeline=state.pipeline||{};
    const pipelineRunning=pipeline.status==='running';
    const pipelineBtn=target.querySelector('.pipeline-run-btn');
    if(pipelineBtn){
      pipelineBtn.disabled=pipelineRunning;
      const stageLabels={'scraper':'Scraping','fast_path':'Ingestando','reporter_fast':'Reporte rápido','reporter_full':'Reporte','parallel':'Generando reporte'};
      const pct=pipeline.percent!=null?pipeline.percent:0;
      if(pipelineRunning){
        const stage=stageLabels[pipeline.stage]||pipeline.stage||'';
        // Show live counts during scraping (percent stays 0)
        if(pipeline.stage==='scraper'){
          pipelineBtn.textContent=`Scraping… ${format(scrapedFiles)} docs`;
        }else if(pipeline.stage==='fast_path'){
          pipelineBtn.textContent=`Indexando… ${format(stagingChunks)} chunks · ${pct}%`;
        }else{
          pipelineBtn.textContent=`${stage}… ${pct}%`;
        }
      }else if(pipeline.status==='done'){
        pipelineBtn.textContent='Pipeline completado ✓';
      }else if(pipeline.status==='failed'){
        pipelineBtn.textContent='Pipeline falló — reintentar';
      }else{
        pipelineBtn.textContent='Ejecutar pipeline completo';
      }
    }
    // Update per-source days inputs only if value changed from server (not user edit)
    // Use simple iteration instead of CSS.escape which may fail with URLs
    const inputRows=target.querySelectorAll('.source-days-table tbody tr');
    effective.forEach((s,i)=>{
      if(i>=inputRows.length)return;
      const inp=inputRows[i].querySelector('input');
      if(inp&&document.activeElement!==inp){
        const serverVal=String(s.days_back??0);
        if(inp.value!==serverVal)inp.value=serverVal;
      }
    });
    // Restore scroll position
    if(sourceTable)sourceTable.scrollTop=tableScrollTop;
    return;
  }

  // First render — build full HTML
  // Scraper section
  let html='<article class="card"><div class="card-head"><div><span class="eyebrow">SCRAPING</span><h3>Adquisición de contenido</h3></div>';
  html+=`<span class="scraper-status ${statusClass(scraperJob.status)}">${scraperRunning?'ejecutando':scraperJob.status==='failed'||scraperJob.status==='error'?'falló':scraperJob.status==='done'?'completado':'inactivo'}</span>`;
  html+='</div>';

  // Unified period selector: mode toggle + dynamic inputs
  const daysChecked=_periodMode==='days'?'checked':'';
  const rangeChecked=_periodMode==='range'?'checked':'';
  const daysDisplay=_periodMode==='days'?'':'none';
  const rangeDisplay=_periodMode==='range'?'':'none';
  const sourceDisplay=_periodMode==='days'?'':'none';
  html+=`<div class="exec-period-selector">
    <div class="period-mode-toggle">
      <label class="period-mode-option">
        <input type="radio" name="period-mode" value="days" ${daysChecked} onchange="switchPeriodMode('days')"> <span>Días hacia atrás</span>
      </label>
      <label class="period-mode-option">
        <input type="radio" name="period-mode" value="range" ${rangeChecked} onchange="switchPeriodMode('range')"> <span>Rango de fechas</span>
      </label>
    </div>
    <div id="period-input-days" class="period-inputs" style="display:${daysDisplay}">
      <label>Días hacia atrás (global): </label><input id="exec-days-global" type="number" min="0" max="365" value="${avgDays}" style="width:80px"><button class="ghost small" onclick="updateAllDays()">Aplicar a todas</button>
    </div>
    <div id="period-input-range" class="period-inputs" style="display:${rangeDisplay}">
      <label>Desde: </label><input id="exec-period-start" type="date" style="width:140px">
      <span class="muted">→</span>
      <label>Hasta: </label><input id="exec-period-end" type="date" style="width:140px">
    </div>
  </div>`;

  // Per-source days editor (only relevant in days mode)
  if(effective.length){
    html+=`<div class="exec-source-days" style="display:${sourceDisplay}"><table class="source-days-table"><thead><tr><th>Fuente</th><th>Estado</th><th>Días</th></tr></thead><tbody>`;
    for(const s of effective){
      const encoded=encodeURIComponent(s.url||'');
      html+=`<tr data-source-url="${esc(s.url||'')}"><td class="source-url-cell">${esc(s.url||'')}</td><td><span class="${s.active?'ok':'bad'}">${s.active?'Activa':'Off'}</span></td><td><input type="number" min="0" max="365" value="${s.days_back??0}" onchange="updateSourceDays('${encoded}',this.value)" style="width:70px"></td></tr>`;
    }
    html+='</tbody></table></div>';
  }

  // Pipeline run button (scraper → fast reporter → full reporter)
  const pipeline=state.pipeline||{};
  const pipelineRunning=pipeline.status==='running';
  const pipelineStage=pipeline.stage||'';
  const pipelinePct=pipeline.percent!=null?pipeline.percent:0;
  let pipelineLabel='Ejecutar pipeline completo';
  if(pipelineRunning){
    const stageLabels={'scraper':'Scraping','fast_path':'Ingestando (BM25 + LanceDB)','reporter_fast':'Reporte rápido','reporter_full':'Reporte (BGE-M3 + Qwen)','parallel':'Generando reporte (BGE-M3 + Qwen)'};
    pipelineLabel=`${stageLabels[pipelineStage]||pipelineStage}… ${pipelinePct}%`;
  }else if(pipeline.status==='done'){
    pipelineLabel='Pipeline completado ✓';
  }else if(pipeline.status==='failed'){
    pipelineLabel='Pipeline falló — reintentar';
  }
  html+=`<div class="actions"><button class="primary pipeline-run-btn" onclick="runPipeline()" ${pipelineRunning?'disabled':''}>${pipelineLabel}</button></div>`;
  html+='</article>';

  // Reporter section removed — merged into pipeline button above

  target.innerHTML=html;
}

async function updateAllDays(){
  const input=$('#exec-days-global');
  if(!input)return;
  const days=Number(input.value);
  if(isNaN(days)||days<0||days>365){toast('Días inválidos (0-365)');return}
  try{
    await api('/api/sources',{method:'POST',body:JSON.stringify({action:'update_days_all',days_back:days})});
    toast('Días actualizados a '+days+' para todas las fuentes');
    // Update per-source inputs in-place without rebuilding the panel
    const rows=document.querySelectorAll('.source-days-table tbody tr');
    rows.forEach(row=>{const inp=row.querySelector('input');if(inp&&document.activeElement!==inp)inp.value=String(days)});
  }catch(e){toast(e.message)}
}

async function updateSourceDays(encodedUrl,days){
  const url=decodeURIComponent(encodedUrl);
  const d=Number(days);
  if(isNaN(d)||d<0||d>365){toast('Días inválidos (0-365)');return}
  try{
    await api('/api/sources',{method:'POST',body:JSON.stringify({action:'update_days',url,days_back:d})});
    toast('Días actualizados para '+url.substring(0,40));
  }catch(e){toast(e.message)}
}

function renderHistory(){
  const history=state.reporter?.history||[];
  const target=$('#report-history');
  if(!target)return;
  if(!history.length){target.innerHTML='';return}
  const currentPath=selectedReportPath||state.reporter?.report?.path;
  const activeIdx=history.findIndex(item=>item.path===currentPath);
  const activeItem=activeIdx>=0?history[activeIdx]:null;
  // Collapsible: show only the active report + a toggle to expand/collapse the rest
  const isExpanded=target.dataset.expanded==='1';
  const items=history.map((item,i)=>{
    const isActive=item.path===currentPath;
    const rel=fmtRelative(item.updated_at||item.generated_at);
    const cls=isActive?' active':'';
    return `<div class="history-row${cls}" onclick="loadReport('${encodeURIComponent(item.path)}')"><div><strong>${esc(item.period?.label||'sin período')}</strong> <span class="muted">${item.category_count} tópicos · ${esc(item.status||'')} · ${rel}</span></div><div style="display:flex;gap:6px;align-items:center"><button class="danger" style="padding:3px 8px;font-size:11px" onclick="event.stopPropagation();deleteReport('${encodeURIComponent(item.path)}')">Borrar</button><span class="${statusClass(item.status)}">${isActive?'● actual':'cargar'}</span></div></div>`;
  }).join('');
  const summary=activeItem
    ?`<strong>${esc(activeItem.period?.label||'sin período')}</strong> <span class="muted">${activeItem.category_count} tópicos · ${esc(activeItem.status||'')} · ${fmtRelative(activeItem.updated_at||activeItem.generated_at)}</span>`
    :`<span class="muted">${history.length} informes disponibles</span>`;
  target.innerHTML=`<div class="history-collapsible">
    <div class="history-header" onclick="toggleHistoryExpand()">
      <div>${summary}</div>
      <span class="history-toggle">${isExpanded?'▲':'▼'} ${history.length} informes</span>
    </div>
    <div class="history-list" style="${isExpanded?'':'display:none'}">${items}</div>
  </div>`;
}

function toggleHistoryExpand(){
  const target=$('#report-history');
  if(!target)return;
  target.dataset.expanded=target.dataset.expanded==='1'?'0':'1';
  renderHistory();
}

async function loadReport(path, opts={}){
  const decoded=decodeURIComponent(path);
  selectedReportPath=decoded;
  try{
    const data=await api('/api/report?path='+encodeURIComponent(decoded));
    selectedReport=data.report;
    // Preserve open <details> state before re-render
    const openDetails=new Set();
    document.querySelectorAll('#categories details').forEach((d,idx)=>{
      if(d.hasAttribute('open'))openDetails.add(idx);
    });
    renderCategories(data.report);
    if(openDetails.size){
      document.querySelectorAll('#categories details').forEach((d,idx)=>{
        if(openDetails.has(idx))d.setAttribute('open','');
      });
    }
    renderHistory();
    renderTopicReview();
    // Scroll to top of reporter panel only on explicit user action
    if(opts.scroll!==false){
      $('#report-meta').scrollIntoView({behavior:'smooth',block:'start'});
    }
  }catch(e){toast(e.message)}
}

function renderCategories(report){
  // Preserve open <details> state across re-renders (by index)
  const openDetails=new Set();
  document.querySelectorAll('#categories details').forEach((d,idx)=>{
    if(d.hasAttribute('open'))openDetails.add(idx);
  });
  if(!report){
    // Show live reporter progress if available
    const rp=state.reporter_progress||{};
    const rpStatus=rp.status||'';
    if(rpStatus==='running'||rpStatus==='failed'||(rp.stage&&rp.percent!=null)){
      const pct=rp.percent||0;
      const stage=rp.stage||'';
      const detail=rp.detail||'';
      const err=rp.error||'';
      const bar=`<div style="height:8px;background:#1a2030;border-radius:4px;overflow:hidden;margin:8px 0"><div style="height:100%;width:${pct}%;background:linear-gradient(90deg,#56d6b0,#78a9ff);transition:width .5s"></div></div>`;
      const cls=rpStatus==='failed'?'bad':(rpStatus==='running'?'ok':'warning');
      $('#report-meta').innerHTML=`<div class="pipeline-live"><span class="eyebrow">REPORTER EN VIVO</span><h3>${esc(stage)} · ${pct}%</h3>${bar}<p class="muted">${esc(detail)}</p>${err?`<p class="bad">Error: ${esc(err)}</p>`:''}</div>`;
    }else{
      $('#report-meta').innerHTML='<div class="notice">No hay un reporte generado todavía. Ejecutá el Reporter desde el panel Resumen.</div>';
    }
    $('#categories').innerHTML='';
    return;
  }
  const gen=report.generation||{};
  const genAt=gen.generated_at||report.updated_at;
  const curation=report.curation_summary||{};
  const totalDocs=Object.values(curation).reduce((a,b)=>a+(Number(b)||0),0);
  const cats=report.categories||[];
  const review=state.reporter?.review||null;
  const reviewBadge=review?`<span class="badge ${statusClass(review.status)}">revisión: ${esc(review.status)}</span>`:'';
  const corpus=report.path?report.path.replace(/\\report\.json$/,'\\corpus'):'';
  const statusCls=statusClass(report.status);

  $('#report-meta').innerHTML=`
    <div class="report-meta-grid">
      <div class="kv"><span class="muted">Período</span><strong>${esc(report.period?.label||'—')}</strong></div>
      <div class="kv"><span class="muted">Generado</span><strong>${fmtDate(genAt)}</strong></div>
      <div class="kv"><span class="muted">Tópicos</span><strong>${format(cats.length)}</strong></div>
      <div class="kv"><span class="muted">Documentos</span><strong>${format(totalDocs)}</strong></div>
      <div class="kv"><span class="muted">Estado</span><strong class="${statusCls}">${esc(report.status||'draft')}</strong></div>
      <div class="kv"><span class="muted">Modelo</span><strong>${esc(gen.model_fingerprint||'—')}</strong></div>
    </div>
    ${Object.keys(curation).length?`<div class="curation-bar"><span class="eyebrow">CURACIÓN</span>${Object.entries(curation).map(([k,v])=>`<span class="badge">${esc(k)}: ${format(v)}</span>`).join('')}</div>`:''}
    <div class="report-actions">
      <button class="approve-btn" onclick="reviewReport('approved')">Aprobar</button>
      <button class="reject-btn" onclick="reviewReport('rejected')">Denegar</button>
      ${selectedReportPath?`<button class="ghost" onclick="clearSelectedReport()">Volver al último</button>`:''}
    </div>`;
  // Parent categories with subtopics
  const parents=report.parent_categories||[];
  const catsById={};
  cats.forEach(c=>catsById[c.category_id]=c);
  if(parents.length){
    $('#categories').innerHTML=parents.map((p,pi)=>{
      const subs=(p.subtopic_ids||[]).map(sid=>catsById[sid]).filter(Boolean);
      const subsHtml=subs.map((s,si)=>`<div class="subtopic" onclick="deepDive('${esc(s.category_id)}')">
        <strong>${esc(s.label)}</strong>
        <span class="muted">${format(s.document_count)} docs · cohesión ${Number(s.cohesion).toFixed(2)}</span>
        <p class="muted" style="margin:2px 0;font-size:12px">${esc(s.description||'')}</p>
      </div>`).join('');
      return `<article class="category parent-category">
        <span class="eyebrow">CATEGORÍA ${String(pi+1).padStart(2,'0')} · ${format(p.document_count)} docs total</span>
        <h3>${esc(p.label)}</h3>
        <p>${esc(p.description)}</p>
        <div class="badges">
          <span class="badge">${format(subs.length)} subtópicos</span>
          <span class="badge">${format(p.document_count)} docs</span>
          <span class="badge">importancia ${Number(p.importance).toFixed(2)}</span>
        </div>
        <details class="subtopics-list"><summary>Ver ${subs.length} subtópicos</summary><div class="subtopics-grid">${subsHtml}</div></details>
        <div class="category-actions">
          <button class="ghost small" onclick="deepDive('${esc(p.category_id)}')">Profundizar categoría</button>
        </div>
      </article>`;
    }).join('');
  }else{
    $('#categories').innerHTML=cats.map((c,i)=>`<article class="category">
      <span class="eyebrow">TOPIC ${String(i+1).padStart(2,'0')} · ${esc(c.evolution)}</span>
      <h3>${esc(c.label)}</h3>
      <p>${esc(c.description)}</p>
      <div class="badges">
        <span class="badge">${format(c.document_count)} docs</span>
        <span class="badge">cohesión ${Number(c.cohesion).toFixed(2)}</span>
        <span class="badge">importancia ${Number(c.importance).toFixed(2)}</span>
        <span class="badge">novedad ${Number(c.novelty).toFixed(2)}</span>
      </div>
      <div class="category-actions">
        <button class="ghost small" onclick="deepDive('${esc(c.category_id)}')">Profundizar</button>
        <button class="ghost" onclick="renameTopic('${encodeURIComponent(c.category_id)}')">Renombrar</button>
      </div>
    </article>`).join('')||'<div class="notice">No se detectaron categorías.</div>';
  }
  // Restore open <details> state by index
  if(openDetails.size){
    document.querySelectorAll('#categories details').forEach((d,idx)=>{
      if(openDetails.has(idx))d.setAttribute('open','');
    });
  }
}

function clearSelectedReport(){
  selectedReport=null;
  selectedReportPath=null;
  renderCategories(state.reporter?.report);
  renderHistory();
  renderTopicReview();
}

// Topic review: singleton/low-doc topics that need human attention
const expandedTopics=new Set();
async function renderTopicReview(){
  const target=$('#topic-review');
  if(!target)return;
  const report=selectedReport||state.reporter?.report;
  const cats=(report?.categories||[]);
  // Candidates: singletons or pending-label topics
  const candidates=cats.filter(c=>c.document_count<2||c.label==='Tema pendiente de nombrar'||c.label?.startsWith('Tópico agrupado'));
  if(!candidates.length){target.innerHTML='';return}
  target.innerHTML='<h3>Candidatos para revisión <span class="muted">('+candidates.length+')</span></h3>'+
    '<p class="muted" style="margin:-8px 0 12px">Tópicos con un solo documento o etiqueta genérica. Clickeá para ver evidencia y decidir.</p>'+
    candidates.map(c=>{
      const isExp=expandedTopics.has(c.category_id);
      return `<div class="topic-candidate${isExp?' expanded':''}" id="tc-${esc(c.category_id)}">
        <div class="tc-header" onclick="toggleTopicCandidate('${esc(c.category_id)}')">
          <div class="tc-info">
            <strong>${esc(c.label)}</strong>
            <span class="muted">${format(c.document_count)} doc · cohesión ${Number(c.cohesion).toFixed(2)} · importancia ${Number(c.importance).toFixed(2)}</span>
          </div>
          <span class="tc-toggle">${isExp?'▼':'▶'}</span>
        </div>
        <div class="tc-body" id="tc-body-${esc(c.category_id)}" style="display:${isExp?'block':'none'}">
          ${c.description?`<p class="tc-desc">${esc(c.description)}</p>`:''}
          <div class="tc-docs" id="tc-docs-${esc(c.category_id)}"><span class="muted">Cargando documentos...</span></div>
          <div class="tc-actions">
            <button class="ghost small" onclick="event.stopPropagation();deepDive('${esc(c.category_id)}')">Profundizar</button>
            <button class="ghost small" onclick="event.stopPropagation();renameTopic('${encodeURIComponent(c.category_id)}')">Renombrar</button>
          </div>
        </div>
      </div>`;
    }).join('');
  // Load docs for expanded candidates
  candidates.filter(c=>expandedTopics.has(c.category_id)).forEach(c=>loadTopicDocs(c.category_id));
}

function toggleTopicCandidate(id){
  const row=$('#tc-'+id);
  const body=$('#tc-body-'+id);
  const toggle=row?.querySelector('.tc-toggle');
  if(!row||!body)return;
  if(expandedTopics.has(id)){
    expandedTopics.delete(id);
    body.style.display='none';
    row.classList.remove('expanded');
    if(toggle)toggle.textContent='▶';
  }else{
    expandedTopics.add(id);
    body.style.display='block';
    row.classList.add('expanded');
    if(toggle)toggle.textContent='▼';
    loadTopicDocs(id);
  }
}

async function loadTopicDocs(categoryId){
  const target=$('#tc-docs-'+categoryId);
  if(!target)return;
  try{
    const data=await api('/api/topic?category_id='+encodeURIComponent(categoryId));
    const docs=data.documents||[];
    if(!docs.length){target.innerHTML='<span class="muted">Sin documentos asociados.</span>';return}
    target.innerHTML='<span class="eyebrow">EVIDENCIA</span>'+docs.map(d=>{
      const path=d.original_path?encodeURIComponent(d.original_path):'';
      const docLink=path?`<button class="ghost small" onclick="event.stopPropagation();viewDoc('${path}')">Abrir</button>`:'';
      const extLink=d.source_url?`<a href="${esc(d.source_url)}" target="_blank" class="ghost small">Fuente</a>`:'';
      return `<div class="tc-doc"><div class="tc-doc-main"><div class="tc-doc-title">${esc(d.title||d.document_id)}</div><span class="muted">${esc(d.source_domain||'')} · ${esc(d.published_at||'fecha desconocida')} · ${esc(d.published_at_confidence||'')}</span></div><div class="tc-doc-actions">${docLink}${extLink}</div></div>`;
    }).join('');
  }catch(e){target.innerHTML=`<span class="bad">${esc(e.message)}</span>`}
}

let _decisionFilter='pending'; // 'pending' | 'promoted' | 'discarded' | 'all'
let _cachedDecisions=[];

async function loadReviewContent(){
  const select=$('#review-corpus');
  const mode=select?select.value:'decisions';
  const decEl=$('#decisions');
  const docEl=$('#review-documents');
  if(!decEl)return;
  if(mode==='decisions'){
    // Show decisions, hide documents
    docEl.innerHTML='';
    try{
      const ds=await api('/api/decisions');
      _cachedDecisions=ds;
      renderDecisions(decEl,ds,_decisionFilter);
    }catch(e){decEl.innerHTML=`<div class="notice">${esc(e.message)}</div>`}
  }else{
    // Show documents, hide decisions
    decEl.innerHTML='';
    try{
      const docs=await api('/api/documents?corpus='+mode+'&limit=200');
      docEl.innerHTML=docs.map(d=>`<div class="doc-row" onclick="viewDoc('${encodeURIComponent(d.original_path||'')}')"><div class="doc-main"><div class="doc-title">${esc(d.title||d.document_id)}</div><span class="muted">${esc(d.source_domain||'')} · ${esc(d.published_at||'fecha desconocida')} · ${esc(d.mime_type||'')}</span></div><span class="score">${d.quality_score==null?'—':Number(d.quality_score).toFixed(2)}</span></div>`).join('')||'<div class="notice">No hay documentos disponibles.</div>';
    }catch(e){docEl.innerHTML=`<div class="notice">${esc(e.message)}</div>`}
  }
}

function filterDecisions(filter){
  _decisionFilter=filter;
  renderDecisions($('#decisions'),_cachedDecisions,filter);
}

function renderDecisions(decEl,ds,filter){
  const pending=ds.filter(d=>['pending','changes_requested'].includes(d.review_status||'pending'));
  const promoted=ds.filter(d=>d.review_status==='approved');
  const discarded=ds.filter(d=>['rejected'].includes(d.review_status)||['duplicate','irrelevant','defer','insufficient_evidence'].includes(d.decision));
  let shown;
  if(filter==='promoted')shown=promoted;
  else if(filter==='discarded')shown=discarded;
  else if(filter==='all')shown=ds;
  else shown=pending; // default: pending
  const activeCls=(f)=>f===filter?' active':'';
  const triage=`<div class="curation-bar">`+
    `<button class="badge${activeCls('pending')}" onclick="filterDecisions('pending')">${pending.length} para revisar</button>`+
    `<button class="badge${activeCls('promoted')}" onclick="filterDecisions('promoted')">${promoted.length} promovidos</button>`+
    `<button class="badge${activeCls('discarded')}" onclick="filterDecisions('discarded')">${discarded.length} descartados/diferidos</button>`+
    `<button class="badge${activeCls('all')}" onclick="filterDecisions('all')">total ${ds.length}</button>`+
    `</div>`;
  const rows=shown.map(d=>{
    const isPending=['pending','changes_requested'].includes(d.review_status||'pending');
    const actions=isPending
      ?`<button class="approve-btn" onclick="review('${encodeURIComponent(d.decision_id)}','approved')">Aprobar</button><button class="reject-btn" onclick="review('${encodeURIComponent(d.decision_id)}','rejected')">Denegar</button>`
      :`<span class="muted">${esc(d.review_status||'')}</span>`;
    return `<div class="decision-row"><div class="decision-main"><strong>${esc(d.title||d.document_id)}</strong><div class="muted">${esc(d.reason||'')} · ${esc(d.review_status||'pending')}</div><div class="score">score promoción ${Number(d.promotion_score||0).toFixed(2)} · relevancia ${Number(d.scores?.relevance||0).toFixed(2)} · novedad ${Number(d.scores?.novelty||0).toFixed(2)} · impacto ${Number(d.scores?.impact||0).toFixed(2)} · calidad ${Number(d.scores?.source_quality||0).toFixed(2)}</div></div><div class="decision-actions">${actions}<button class="ghost small" onclick="viewDoc('${encodeURIComponent(d.original_path||'')}')">Ver</button></div></div>`;
  }).join('');
  decEl.innerHTML=triage+(rows||'<div class="notice">No hay decisiones en esta categoría.</div>');
}

function viewDoc(path){
  if(!path)return toast('Este documento no tiene path visualizable');
  const decoded=decodeURIComponent(path);
  $('#viewer-title').textContent=decoded.split(/[\\/]/).pop();
  const mime=decoded.toLowerCase().endsWith('.pdf')?'application/pdf':decoded.toLowerCase().endsWith('.html')?'text/html':'text/plain';
  if(mime==='application/pdf')$('#viewer-body').innerHTML=`<iframe src="/api/document/raw?path=${encodeURIComponent(decoded)}"></iframe>`;
  else api('/api/document?path='+encodeURIComponent(decoded)).then(d=>{const text=d.mime_type==='application/json'?JSON.stringify(JSON.parse(atob(d.content_base64)),null,2):new TextDecoder().decode(Uint8Array.from(atob(d.content_base64),c=>c.charCodeAt(0)));$('#viewer-body').textContent=text}).catch(e=>$('#viewer-body').textContent=e.message);
  viewer.showModal();
}

async function addSource(e){e.preventDefault();try{await api('/api/sources',{method:'POST',body:JSON.stringify({action:'add',url:$('#source-url').value,days_back:Number($('#source-days').value)})});$('#source-url').value='';toast('Fuente agregada');refreshAll()}catch(err){toast(err.message)}}
async function toggleSource(encoded,active,isBase){try{await api('/api/sources',{method:'POST',body:JSON.stringify({action:active?'disable':'enable',url:decodeURIComponent(encoded)})});toast(active?'Fuente desactivada':'Fuente activada');refreshAll()}catch(e){toast(e.message)}}
async function runScraper(){try{await api('/api/scraper/run',{method:'POST',body:'{}'});toast('Scraper iniciado');refreshAll()}catch(e){toast(e.message)}}
async function runReporter(embeddings,llm){try{await api('/api/reporter/run',{method:'POST',body:JSON.stringify({period:'2026-08',embeddings,llm})});toast('Reporter iniciado');refreshAll()}catch(e){toast(e.message)}}

function switchPeriodMode(mode){
  _periodMode=mode;
  const daysDiv=$('#period-input-days');
  const rangeDiv=$('#period-input-range');
  const sourceTable=document.querySelector('.exec-source-days');
  if(mode==='range'){
    if(daysDiv)daysDiv.style.display='none';
    if(rangeDiv)rangeDiv.style.display='';
    if(sourceTable)sourceTable.style.display='none';
  }else{
    if(daysDiv)daysDiv.style.display='';
    if(rangeDiv)rangeDiv.style.display='none';
    if(sourceTable)sourceTable.style.display='';
  }
}

async function runPipeline(){
  try{
    const modeEl=document.querySelector('input[name="period-mode"]:checked');
    const mode=modeEl?modeEl.value:'days';
    const payload={period_mode:mode};
    if(mode==='range'){
      const startInput=$('#exec-period-start');
      const endInput=$('#exec-period-end');
      if(startInput&&startInput.value)payload.period_start=startInput.value+'T00:00:00Z';
      if(endInput&&endInput.value)payload.period_end=endInput.value+'T23:59:59Z';
      if(!payload.period_start||!payload.period_end){toast('Indicá ambas fechas del rango');return}
    }else{
      const daysInput=$('#exec-days-global');
      if(daysInput){
        const days=Number(daysInput.value);
        if(!isNaN(days)&&days>=0)payload.days_back=days;
      }
    }
    await api('/api/pipeline/run',{method:'POST',body:JSON.stringify(payload)});
    toast('Pipeline iniciado: scraper → reporte rápido → reporte completo');
    // Switch to Reporter tab immediately
    document.querySelectorAll('.tabs .tab').forEach(btn=>btn.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(panel=>panel.classList.remove('active'));
    const reporterTab=document.querySelector('.tabs .tab[data-panel="reporter"]');
    if(reporterTab)reporterTab.classList.add('active');
    const reporterPanel=$('#panel-reporter');
    if(reporterPanel)reporterPanel.classList.add('active');
    refreshAll();
  }catch(e){toast(e.message)}
}
async function reviewReport(status){
  const path=selectedReportPath||state.reporter?.report?.path;
  if(!path){toast('No hay reporte seleccionado');return}
  try{
    const res=await api('/api/reports/review',{method:'POST',body:JSON.stringify({status,path,decided_by:'web-user'})});
    if(status==='approved'){
      toast(`Aprobado: ${res.promoted_docs||0} docs, ${res.promoted_chunks||0} chunks, ${res.promoted_vectors||0} vectores, ${res.archived_files||0} archivos archivados`);
      selectedReportPath=null; // reset selection since report is now promoted
    }else if(status==='rejected'){
      toast('Reporte denegado y eliminado completamente');
      selectedReportPath=null;
    }else{
      toast('Estado del corpus actualizado');
    }
    refreshAll();
  }catch(e){toast(e.message)}
}

async function deleteReport(encodedPath){
  const path=decodeURIComponent(encodedPath);
  if(!confirm('¿Borrar este reporte? Se eliminarán: reporte, corpus, índices (BM25 + Tantivy + LanceDB) y decisiones de curación.'))return;
  try{
    await api('/api/reports/delete',{method:'POST',body:JSON.stringify({path})});
    toast('Reporte borrado');
    if(selectedReportPath===path)selectedReportPath=null;
    refreshAll();
  }catch(e){toast(e.message)}
}
async function review(id,status){try{await api('/api/decisions/review',{method:'POST',body:JSON.stringify({decision_id:decodeURIComponent(id),status,decided_by:'web-user'})});toast('Decisión actualizada');loadReviewContent()}catch(e){toast(e.message)}}

function deepDive(categoryId){
  const report=selectedReport||state.reporter?.report;
  const category=(report?.categories||[]).find(item=>item.category_id===categoryId);
  const topic=category?.label||'este tema';
  const corpus=report?.path?report.path.replace(/\\report\.json$/,'\\corpus'):'';
  const searchContext=[category?.label,category?.description,(category?.subtopics||[]).join(' ')].filter(Boolean).join(' ');
  const defaultQ=`Explica los puntos principales y la evidencia sobre ${topic}.`;
  // Deep dive consolidado en el chat: mismo agente, mismo tool loop, misma
  // memoria — el retrieval corre sobre el corpus del reporte server-side.
  if(!corpus){toast('Este reporte no tiene corpus asociado');return}
  ddContext={corpus,category_id:categoryId,search:searchContext,topic};
  renderDdChip();
  document.querySelectorAll('.nav-item,.panel').forEach(x=>x.classList.remove('active'));
  const nav=document.querySelector('.nav-item[data-panel="agent"]');
  if(nav)nav.classList.add('active');
  const panel=$('#panel-agent');
  if(panel)panel.classList.add('active');
  loadAgentPanel();
  const input=$('#agent-chat-input');
  if(input){input.value=defaultQ;input.focus()}
}

function renderDdChip(){
  const chip=$('#dd-chip');
  if(!chip)return;
  if(!ddContext){chip.style.display='none';return}
  chip.style.display='flex';
  chip.innerHTML=`<span class="dd-label">Profundizando: ${esc(ddContext.topic)}</span><button type="button" class="dd-close" onclick="clearDdContext()" title="Salir del modo deep dive">×</button>`;
}

function clearDdContext(){
  ddContext=null;
  renderDdChip();
}

async function renameTopic(encodedId){
  const categoryId=decodeURIComponent(encodedId);
  const label=window.prompt('Nombre del tópico:');
  if(!label)return;
  try{await api('/api/topics/edit',{method:'POST',body:JSON.stringify({category_id:categoryId,label})});toast('Tópico actualizado');if(selectedReportPath){await loadReport(selectedReportPath)}else{refreshAll()}}catch(error){toast(error.message)}
}

// Sidebar colapsable: rail de íconos (56px). Estado persistido.
function toggleSidebar(){
  const sb=document.getElementById('sidebar');
  if(!sb)return;
  const collapsed=sb.classList.toggle('collapsed');
  const t=document.getElementById('sidebar-toggle');
  if(t)t.textContent=collapsed?'›':'‹';
  try{localStorage.setItem('ipa_sidebar_collapsed',collapsed?'1':'0')}catch(e){}
}

// Panel de sesiones redimensionable: drag handle, ancho persistido.
(function initSessResizer(){
  const aside=document.querySelector('.agent-sessions-panel');
  const layout=document.querySelector('.agent-layout');
  const handle=document.getElementById('sess-resizer');
  if(!aside||!layout||!handle)return;
  try{
    const w=parseInt(localStorage.getItem('ipa_sess_w')||'',10);
    if(w>=160&&w<=560)layout.style.setProperty('--sess-w',w+'px');
  }catch(e){}
  handle.addEventListener('mousedown',e=>{
    e.preventDefault();
    const startX=e.clientX,startW=aside.getBoundingClientRect().width;
    handle.classList.add('dragging');
    document.body.style.userSelect='none';
    function mv(ev){
      const w=Math.min(560,Math.max(160,startW+ev.clientX-startX));
      layout.style.setProperty('--sess-w',w+'px');
      try{localStorage.setItem('ipa_sess_w',String(Math.round(w)))}catch(e2){}
    }
    function up(){
      handle.classList.remove('dragging');
      document.body.style.userSelect='';
      document.removeEventListener('mousemove',mv);
      document.removeEventListener('mouseup',up);
    }
    document.addEventListener('mousemove',mv);
    document.addEventListener('mouseup',up);
  });
})();

// Topbar ocultable: queda un handle flotante (▾) para restaurarla.
function toggleTopbar(){
  const hidden=document.body.classList.toggle('no-topbar');
  const r=document.getElementById('topbar-restore');
  if(r)r.style.display=hidden?'block':'none';
  try{localStorage.setItem('ipa_topbar_hidden',hidden?'1':'0')}catch(e){}
}

(function initChrome(){
  try{
    if(localStorage.getItem('ipa_sidebar_collapsed')==='1'){
      const sb=document.getElementById('sidebar');
      if(sb)sb.classList.add('collapsed');
      const t=document.getElementById('sidebar-toggle');
      if(t)t.textContent='›';
    }
    if(localStorage.getItem('ipa_topbar_hidden')==='1'){
      document.body.classList.add('no-topbar');
      const r=document.getElementById('topbar-restore');
      if(r)r.style.display='block';
    }
  }catch(e){}
})();

document.querySelectorAll('.nav-item').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.nav-item,.panel').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  $('#panel-'+b.dataset.panel).classList.add('active');
  if(b.dataset.panel==='approvals')loadApprovals();
  if(b.dataset.panel==='agent')loadAgentPanel();
  if(b.dataset.panel==='knowledge')loadKnowledge();
});

// ── Agent panel: chat + sesiones ─────────────────────────────────────────
let agentSessionId=null;
let chatRole='general';
let ddContext=null;
function setChatRole(role){
  chatRole=role;
  document.querySelectorAll('#chat-role .role-btn').forEach(b=>b.classList.toggle('active',b.dataset.role===role));
}

async function loadAgentPanel(){
  try{
    const d=await api('/api/agent/sessions');
    renderSessionList(d.sessions||[]);
    if(!agentSessionId&&d.sessions.length){openAgentSession(d.sessions[0].session_id)}
    else if(!agentSessionId)renderChatEmpty();
  }catch(e){$('#chat-messages').innerHTML=`<div class="notice">${esc(e.message)}</div>`}
  loadTutorRoadmaps();
}

// ── Roadmap Tutor: stepper visual por unidad ─────────────────────────────
async function loadTutorRoadmaps(){
  const el=$('#tutor-roadmaps');
  if(!el)return;
  try{
    const d=await api('/api/tutor/roadmaps');
    renderTutorRoadmaps(d.roadmaps||[]);
  }catch(e){el.innerHTML=`<div class="muted" style="font-size:12px">${esc(e.message)}</div>`}
}

function renderTutorRoadmaps(roadmaps){
  const el=$('#tutor-roadmaps');
  if(!roadmaps.length){el.innerHTML='<div class="muted" style="font-size:12px;padding:6px 0">Sin roadmaps todavía. Activá el rol Tutor y pedí aprender un tema.</div>';return}
  el.innerHTML=roadmaps.map(r=>{
    const steps=r.units.map(u=>{
      const mark=u.status==='done'?'✓':u.status==='current'?'▶':String(u.order);
      return `<div class="rm-step ${esc(u.status)}" title="${esc(u.reason)}">
        <span class="rm-num">${mark}</span>
        <div class="rm-body">
          <div class="rm-title">${esc(u.title)}</div>
          <div class="rm-meta">${u.minutes} min · ${esc((u.assessment_types||[]).join(', '))}</div>
        </div>
      </div>`;
    }).join('');
    const m=r.mastery
      ?`<div class="rm-mastery">mastery ${esc(r.mastery.status)}${r.mastery.score!=null?' · '+(r.mastery.score*100).toFixed(0)+'%':''} · ${r.mastery.attempts} intentos · ${r.mastery.evidence_count} evidencias</div>`
      :'';
    return `<div class="rm-card" onclick="goToTutorChat()" title="Ir al chat del tutor">
      <div class="rm-head"><span class="rm-topic">${esc(r.topic)}</span>
        <span class="rm-head-actions">
          <span class="rm-status ${esc(r.status)}">${esc(r.status)}</span>
          <button class="icon-btn tiny" title="Archivar roadmap" onclick="event.stopPropagation();archiveTutorRoadmap('${esc(r.roadmap_id)}')">📦</button>
        </span>
      </div>
      ${steps}${m}
    </div>`;
  }).join('');
}

async function archiveTutorRoadmap(roadmapId){
  if(!confirm('¿Archivar este roadmap? Sale de la lista; el progreso se conserva.'))return;
  try{
    await api('/api/tutor/roadmap/archive',{method:'POST',body:JSON.stringify({roadmap_id:roadmapId,archived:true})});
    toast('Roadmap archivado');
    loadTutorRoadmaps();
  }catch(e){toast(e.message)}
}

// Click en la card del roadmap → panel Agente, rol Tutor, input listo.
function goToTutorChat(){
  document.querySelectorAll('.nav-item,.panel').forEach(x=>x.classList.remove('active'));
  const nav=document.querySelector('.nav-item[data-panel="agent"]');
  if(nav)nav.classList.add('active');
  const panel=$('#panel-agent');
  if(panel)panel.classList.add('active');
  setChatRole('tutor');
  loadAgentPanel();
  const input=$('#agent-chat-input');
  if(input){input.focus();input.placeholder='Seguimos con el roadmap…'}
}

async function newAgentSession(){
  try{
    const d=await api('/api/agent/sessions/manage',{method:'POST',body:JSON.stringify({action:'new'})});
    agentSessionId=d.session_id;
    clearDdContext();
    $('#chat-messages').innerHTML='<div class="muted" style="text-align:center;padding:40px">Nueva conversación lista. Escribí un mensaje.</div>';
    loadAgentPanel();
  }catch(e){toast(e.message)}
}

async function renameAgentSession(sessionId,currentTitle){
  const title=window.prompt('Nuevo nombre:',currentTitle||'');
  if(!title||!title.trim())return;
  try{
    await api('/api/agent/sessions/manage',{method:'POST',body:JSON.stringify({action:'rename',session_id:sessionId,title:title.trim()})});
    toast('Sesión renombrada');
    loadAgentPanel();
  }catch(e){toast(e.message)}
}

async function archiveAgentSession(sessionId){
  if(!confirm('¿Archivar esta conversación? Los episodios se conservan; sale de la lista activa.'))return;
  try{
    await api('/api/agent/sessions/manage',{method:'POST',body:JSON.stringify({action:'archive',session_id:sessionId})});
    if(agentSessionId===sessionId){agentSessionId=null;$('#chat-messages').innerHTML='<div class="muted" style="text-align:center;padding:40px">Sesión archivada.</div>'}
    toast('Sesión archivada');
    loadAgentPanel();
  }catch(e){toast(e.message)}
}

function renderSessionList(sessions){
  const el=$('#agent-sessions-list');
  if(!el)return;
  if(!sessions.length){el.innerHTML='<div class="muted" style="padding:12px">Sin sesiones aún. Escribí un mensaje para empezar.</div>';return}
  el.innerHTML=sessions.map(s=>`
    <div class="session-item ${s.session_id===agentSessionId?'active':''}" onclick="openAgentSession('${esc(s.session_id)}')">
      <div class="session-row">
        <strong>${esc(s.title||s.session_id.slice(-14))}</strong>
        <span class="session-actions">
          <button class="icon-btn tiny" title="Renombrar" onclick="event.stopPropagation();renameAgentSession('${esc(s.session_id)}','${esc((s.title||'').replace(/'/g,"\\'"))}')">✏️</button>
          <button class="icon-btn tiny" title="Archivar" onclick="event.stopPropagation();archiveAgentSession('${esc(s.session_id)}')">📦</button>
        </span>
      </div>
      <div class="muted" style="font-size:11px">${esc(s.interface||'')} · ${s.episode_count||0} episodios · ${fmtRelative(s.last_active_at)}${s.consolidated_at?` · <span class="sum-flag" title="${esc((s.summary||'').slice(0,240))}">✦ resumida</span>`:''}</div>
    </div>`).join('');
}

function renderChatEmpty(){
  $('#chat-messages').innerHTML='<div class="chat-empty">Escribí un mensaje para empezar a conversar con tu agente.</div>';
  _renderedEpisodeIds=new Set();
}

// Episodios ya renderizados (deduplicación: el poller solo agrega nuevos)
let _renderedEpisodeIds=new Set();

function renderEpisode(e){
  return `<div class="chat-msg ${e.turn_role}" data-episode-id="${esc(e.episode_id)}">${e.turn_role==='assistant'?renderMarkdown(e.content):esc(e.content)}<div class="chat-meta">${esc(e.turn_role)} · ${fmtRelative(e.created_at)}</div></div>`;
}

// Polling: detecta episodios nuevos en la sesión activa (p. ej. el resumen
// proactivo que el agente escribe cuando termina el pipeline). Append-only:
// nunca re-renderiza lo que ya está (evita duplicar mensajes locales).
setInterval(async()=>{
  if(!agentSessionId)return;
  const panel=$('#panel-agent');
  if(!panel||!panel.classList.contains('active'))return;
  try{
    const d=await api('/api/agent/session?session_id='+encodeURIComponent(agentSessionId));
    const msgs=$('#chat-messages');
    const episodes=d.episodes||[];
    const fresh=episodes.filter(e=>!_renderedEpisodeIds.has(e.episode_id));
    if(!fresh.length)return;
    // Si el chat está en medio de un streaming local, no tocar el DOM
    if(msgs.querySelector('.chat-msg.streaming'))return;
    const hadEmpty=msgs.querySelector('.chat-empty');
    for(const e of fresh){
      _renderedEpisodeIds.add(e.episode_id);
      if(hadEmpty)hadEmpty.remove();
      // Dedup: eliminar bubbles locales optimistas (user append / streaming
      // ya cerrado) cuyo episodio canónico acaba de llegar. Sin esto, un
      // re-render fallido dejaba ambas copias visibles para siempre.
      const head=(e.content||'').replace(/\s+/g,' ').trim().slice(0,50);
      if(head){
        msgs.querySelectorAll('.chat-msg:not([data-episode-id]):not(.streaming)').forEach(l=>{
          if(l.classList.contains(e.turn_role)
             && l.textContent.replace(/\s+/g,' ').includes(head)){
            l.remove();
          }
        });
      }
      msgs.insertAdjacentHTML('beforeend',renderEpisode(e));
    }
    scrollChatToBottom();
    const isProactive=fresh.some(e=>e.turn_role==='assistant');
    if(isProactive&&(document.hidden||!$('#panel-agent').classList.contains('active'))){
      toast('💬 Tu agente tiene novedades');
    }
  }catch{}
},10000);

// Card de pipeline en vivo dentro del chat: se crea cuando el agente lanza
// run_pipeline y se actualiza cada 5s desde /api/state hasta que termina.
let _pipelineCardTimer=null;
function mountPipelineCardInChat(){
  const msgs=$('#chat-messages');
  if(!msgs||document.getElementById('chat-pipeline-card'))return;
  const card=document.createElement('div');
  card.id='chat-pipeline-card';
  card.className='chat-pipeline-card';
  card.innerHTML='<span class="eyebrow">PIPELINE EN CURSO</span><div class="pc-stage">Iniciando…</div><div class="pc-bar"><div class="pc-fill" style="width:0%"></div></div><div class="muted" style="font-size:11px" id="chat-pipeline-detail"></div>';
  msgs.appendChild(card);
  scrollChatToBottom();
  if(_pipelineCardTimer)clearInterval(_pipelineCardTimer);
  _pipelineCardTimer=setInterval(async()=>{
    const cardEl=document.getElementById('chat-pipeline-card');
    if(!cardEl){clearInterval(_pipelineCardTimer);_pipelineCardTimer=null;return}
    try{
      const st=await api('/api/state');
      const p=st.pipeline||{};
      const stageLabels={'scraper':'Scraping','fast_path':'Ingestando (BM25 + LanceDB)','reporter_fast':'Reporte rápido','reporter_full':'Reporte (BGE-M3 + Qwen)','parallel':'Generando reporte'};
      if(p.status==='running'){
        cardEl.querySelector('.pc-fill').style.width=(p.percent||0)+'%';
        cardEl.querySelector('.pc-stage').textContent=`${stageLabels[p.stage]||p.stage||'Ejecutando'} · ${p.percent||0}%`;
        const detail=$('#chat-pipeline-detail');
        if(detail)detail.textContent=p.detail||'';
      }else{
        // Terminó (o falló): congelar la card y dejar de poller
        cardEl.querySelector('.pc-fill').style.width='100%';
        cardEl.querySelector('.pc-stage').textContent=p.status==='done'?'✓ Completado':'Estado: '+(p.status||'finalizado');
        clearInterval(_pipelineCardTimer);_pipelineCardTimer=null;
      }
    }catch{}
  },5000);
}

async function openAgentSession(sessionId){
  agentSessionId=sessionId;
  try{
    const d=await api('/api/agent/session?session_id='+encodeURIComponent(sessionId));
    renderSessionList((await api('/agent/sessions'.replace('/agent','/api/agent'))).sessions);
    const msgs=$('#chat-messages');
    _renderedEpisodeIds=new Set((d.episodes||[]).map(e=>e.episode_id));
    if(!d.episodes.length){renderChatEmpty();mountPendingTutorGates();return}
    $('#chat-messages').innerHTML=d.episodes.map(renderEpisode).join('');
    scrollChatToBottom();
    mountPendingTutorGates();
  }catch(e){toast(e.message)}
}

function scrollChatToBottom(){
  const el=$('#chat-messages');
  if(el)el.scrollTop=el.scrollHeight;
}

function sendAgentText(text){
  const input=$('#agent-chat-input');
  input.value=text;
  sendAgentMessage({preventDefault(){}});
}
// Gate del tutor: card con Aprobar / Rechazar / Debatir. Se re-monta desde
// /api/agent/approvals después de cada re-render (los gates no son episodios
// y el re-render canónico los borra).
function tutorGateCard(kind,id,heading,subHtml){
  return `<div class="tutor-gate" id="gate-${esc(id)}" data-gate-kind="${esc(kind)}">
    <div style="width:100%;font-size:12px;color:var(--muted)">${kind==='roadmap'?esc(heading||'Roadmap propuesto'):'Investigación propuesta'}${sub||''}</div>
    <button class="approve" onclick="tutorDecide('${kind}','${esc(id)}','approve')">Aprobar</button>
    <button class="reject" onclick="tutorDecide('${kind}','${esc(id)}','reject')">Rechazar</button>
    ${kind==='roadmap'?`<button class="ghost small" onclick="tutorDebate('${esc(id)}')">Debatir</button>`:''}
  </div>`;
}
function tutorDebate(id){
  const gate=document.getElementById('gate-'+id);
  if(!gate)return;
  if(gate.querySelector('.debate-row'))return;
  const row=document.createElement('div');
  row.className='debate-row';
  row.style.cssText='display:flex;gap:6px;width:100%;margin-top:6px';
  row.innerHTML=`<input type="text" class="debate-input" placeholder="Qué ajustar: orden, profundidad, alcance…" style="flex:1;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:6px 10px;color:var(--text);font-size:12px">
    <button class="primary" style="padding:4px 12px;font-size:12px">Enviar</button>`;
  const send=()=>{
    const txt=row.querySelector('input').value.trim();
    if(!txt)return;
    row.remove();
    sendAgentText(txt);
  };
  row.querySelector('button').onclick=send;
  row.querySelector('input').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();send()}});
  gate.appendChild(row);
  row.querySelector('input').focus();
}
async function mountPendingTutorGates(){
  const msgs=$('#chat-messages');
  if(!msgs||!agentSessionId)return;
  try{
    const d=await api('/api/agent/approvals?status=pending');
    const gates=(d.approvals||[]).filter(a=>
      (a.kind==='roadmap'||a.kind==='research_request')
      &&['pending','proposed','pending_approval'].includes(a.status));
    for(const a of gates){
      if(document.getElementById('gate-'+a.id))continue;
      const sub=`<div class="muted" style="font-size:12px;margin-top:4px">${esc(a.summary||'')}</div>`;
      msgs.insertAdjacentHTML('beforeend',tutorGateCard(a.kind==='roadmap'?'roadmap':'research',a.id,a.kind==='roadmap'?(a.title||'Roadmap propuesto'):'Investigación propuesta',sub));
    }
    scrollChatToBottom();
  }catch{}
}
async function tutorDecide(kind,id,decision){
  const url=kind==='roadmap'?'/api/tutor/roadmap/decision':'/api/tutor/research/decision';
  const body=kind==='roadmap'?{roadmap_id:id,decision}:{request_id:id,decision};
  if(agentSessionId)body.session_id=agentSessionId;
  const gate=document.getElementById('gate-'+id);
  if(gate)gate.querySelectorAll('button').forEach(b=>b.disabled=true);
  try{
    const d=await api(url,{method:'POST',body:JSON.stringify(body)});
    if(!d.ok){toast(d.error||'Error en la decisión');if(gate)gate.querySelectorAll('button').forEach(b=>b.disabled=false);return}
    if(gate){
      const status=d.status==='approved'?'<span style="color:var(--accent)">Aprobado</span>':'<span style="color:var(--danger)">Rechazado</span>';
      gate.innerHTML=`<div style="font-size:12px">${status}</div>`;
    }
    if(kind==='research'&&d.status==='approved'){
      const ind=$('#research-indicator');
      if(ind){ind.style.display='flex';$('#research-indicator-text').textContent='Investigación del tutor en curso…'}
    }
    loadApprovals&&loadApprovals();
    loadTutorRoadmaps();
  }catch(e){
    toast(e.message);
    if(gate)gate.querySelectorAll('button').forEach(b=>b.disabled=false);
  }
}

async function sendAgentMessage(event){
  event.preventDefault();
  const input=$('#agent-chat-input');
  const message=input.value.trim();
  if(!message)return;
  input.value='';
  // Cancelar stream anterior si todavía está generando (bug del 2026-09-08:
  // mandar un mensaje nuevo no cancelaba el anterior, dejando bubbles
  // streaming vacíos superpuestos).
  if(window._agentStreamController){
    try{window._agentStreamController.abort()}catch{}
    window._agentStreamController=null;
  }
  // Remover bubbles "streaming" huérfanos del turno anterior
  document.querySelectorAll('.chat-msg.assistant.streaming').forEach(el=>el.remove());
  appendChatMsg('user',message);
  // Create assistant placeholder for streaming
  const el=$('#chat-messages');
  const assistantDiv=document.createElement('div');
  assistantDiv.className='chat-msg assistant streaming';
  assistantDiv.innerHTML='<div class="chat-content"></div><div class="chat-meta">assistant · ahora</div>';
  el.appendChild(assistantDiv);
  const contentEl=assistantDiv.querySelector('.chat-content');
  el.scrollTop=el.scrollHeight;

  try{
    const body={message,role:chatRole};
    if(agentSessionId)body.session_id=agentSessionId;
    if(ddContext){
      body.context='deep_dive';
      body.corpus=ddContext.corpus;
      body.category_id=ddContext.category_id;
      body.search=ddContext.search;
    }
    const controller=new AbortController();
    window._agentStreamController=controller;
    const response=await fetch('/api/agent/chat/stream',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body),
      signal:controller.signal
    });
    if(!response.ok){
      const errData=await response.json().catch(()=>({error:'HTTP '+response.status}));
      contentEl.textContent='[error] '+errData.error;
      assistantDiv.classList.remove('streaming');
      return;
    }
    const reader=response.body.getReader();
    const decoder=new TextDecoder();
    let buffer='';
    let fullReply='';
    let lastSources=[];
    const pendingTutorGates=[];
    while(true){
      const {done,value}=await reader.read();
      if(done)break;
      buffer+=decoder.decode(value,{stream:true});
      const lines=buffer.split('\n');
      buffer=lines.pop()||'';
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        try{
          const data=JSON.parse(line.slice(6));
          if(data.type==='session'){agentSessionId=data.session_id}
          else if(data.type==='token'){fullReply+=data.text;contentEl.textContent=fullReply;el.scrollTop=el.scrollHeight}
          else if(data.type==='retrieval'){
            const stageLabels={
              'start':'Buscando en el corpus…',
              'searching':'Consultando índice vectorial…',
              'found':`Encontrados ${data.count} chunks relevantes`,
              'empty':'Sin resultados en el corpus',
              'no_corpus':'Corpus no disponible',
              'timeout':'Retrieval demorado — respondiendo sin contexto',
              'error':`Error en retrieval: ${data.error||'?'}`
            };
            if(data.stage==='found'&&data.sources){lastSources=data.sources}
            if(data.stage==='empty'){
              contentEl.innerHTML=`<span class="tool-activity">${esc(stageLabels.empty)}</span>`+
                `<button class="research-btn" data-q="${esc(data.query||'')}" onclick="sendAgentText('investigá '+this.dataset.q)">🔎 Investigar en la web</button>`;
            }else{
              contentEl.innerHTML=`<span class="tool-activity">${esc(stageLabels[data.stage]||data.stage)}</span>`;
            }
            el.scrollTop=el.scrollHeight;
          }
          else if(data.type==='tool_start'){
            fullReply='';
            contentEl.innerHTML=`<span class="tool-activity">🔧 Ejecutando <strong>${esc(data.tool)}</strong>${data.args&&Object.keys(data.args).length?` ${esc(JSON.stringify(data.args))}`:''}…</span>`;
            el.scrollTop=el.scrollHeight;
          }
          else if(data.type==='tool_result'){
            contentEl.innerHTML=`<span class="tool-activity ${data.ok?'ok':'fail'}">${data.ok?'✓':'✗'} ${esc(data.tool)}</span>`;
            el.scrollTop=el.scrollHeight;
            if((data.tool==='run_pipeline'||data.tool==='run_ingestion')&&data.ok)mountPipelineCardInChat();
          }
          else if(data.type==='new_message'){
            // El agente continúa en una burbuja nueva (post-tool)
            fullReply='';
            const div2=document.createElement('div');
            div2.className='chat-msg assistant streaming';
            div2.innerHTML='<div class="chat-content"></div><div class="chat-meta">assistant · ahora</div>';
            el.appendChild(div2);
            contentEl=div2.querySelector('.chat-content');
            el.scrollTop=el.scrollHeight;
          }
          else if(data.type==='roadmap_proposal'){
            pendingTutorGates.push({kind:'roadmap',id:data.roadmap_id,title:data.title,items:data.items});
          }
          else if(data.type==='research_proposal'){
            pendingTutorGates.push({kind:'research',id:data.request_id,query:data.query});
          }
          else if(data.type==='error'){contentEl.textContent='[error] '+(data.error||'unknown');assistantDiv.classList.remove('streaming')}
          else if(data.type==='done'){
            if(data.follow_up){
              contentEl.innerHTML=renderMarkdown(data.follow_up);
            }else{
              fullReply=data.reply||fullReply;
              contentEl.innerHTML=renderMarkdown(fullReply);
            }
            if(lastSources.length){
              const items=lastSources.map(s=>{
                const label=s.source_domain||s.document_id||'?';
                const inner=s.source_url
                  ?`<a href="${esc(s.source_url)}" target="_blank" rel="noopener">[${s.n}] ${esc(label)}</a>`
                  :`[${s.n}] ${esc(label)}`;
                return `<div class="source-item">${inner}</div>`;
              }).join('');
              contentEl.innerHTML+=`<div class="sources-block"><div class="sources-title">Fuentes</div>${items}</div>`;
            }
            if(pendingTutorGates.length){
              const gates=pendingTutorGates.map(g=>{
                const list=g.items?`<ul style="margin:6px 0 0 18px;font-size:12px;color:var(--muted)">${g.items.map(c=>`<li>${esc(c)}</li>`).join('')}</ul>`:'';
                const sub=g.query?`<div class="muted" style="font-size:12px;margin-top:4px">${esc(g.query)}</div>`:'';
                return tutorGateCard(g.kind,g.id,g.title,sub+list);
              }).join('');
              contentEl.innerHTML+=gates;
            }
            assistantDiv.classList.remove('streaming');
          }
        }catch{}
      }
    }
    assistantDiv.classList.remove('streaming');
    // Re-render desde el servidor: la verdad canónica. Elimina duplicados
    // (el append local del user + el episodio del servidor) y muestra los
    // episode_ids reales para que el poller deduplique correctamente.
    // Retry: si el fetch falla, el poller dedup limpia las copias locales
    // en el próximo tick — no dejamos duplicados permanentes.
    let sessData=null;
    for(let a=0;a<3&&!sessData;a++){
      try{sessData=await api('/api/agent/session?session_id='+encodeURIComponent(agentSessionId))}
      catch(e){await new Promise(r=>setTimeout(r,700))}
    }
    if(sessData){
      const msgs=$('#chat-messages');
      _renderedEpisodeIds=new Set((sessData.episodes||[]).map(e=>e.episode_id));
      msgs.innerHTML=sessData.episodes.map(renderEpisode).join('');
      scrollChatToBottom();
    }
    loadAgentPanel();
    mountPendingTutorGates();
  }catch(err){
    if(err.name==='AbortError'){
      // Cancelado por un mensaje nuevo: remover el bubble vacío
      assistantDiv.remove();
    }else{
      contentEl.textContent='[error] '+err.message;
      assistantDiv.classList.remove('streaming');
    }
  }finally{
    window._agentStreamController=null;
  }
}

function appendChatMsg(role,content){
  const el=$('#chat-messages');
  const div=document.createElement('div');
  div.className='chat-msg '+role;
  if(role==='assistant'){div.innerHTML=renderMarkdown(content)}
  else{div.textContent=content}
  el.appendChild(div);
  el.scrollTop=el.scrollHeight;
}

// ── Unified approvals queue (Fase 3) ─────────────────────────────────────
async function loadApprovals(){
  const el=$('#approvals-queue');
  if(!el)return;
  try{
    const filter=$('#approvals-filter')?.value||'pending';
    const d=await api('/api/agent/approvals?status='+encodeURIComponent(filter));
    if(!d.approvals.length){el.innerHTML='<div class="muted" style="text-align:center;padding:40px">No hay nada esperando tu decisión. Todo al día.</div>';return}
    const kindLabels={memory_consolidation:'Consolidación de memoria',mastery_inference:'Inferencia de mastery',roadmap:'Roadmap pedagógico',research_request:'Research request',reporter_decision:'Curación Reporter'};
    el.innerHTML=d.approvals.map(a=>`
      <div class="approval-card kind-${esc(a.kind)}">
        <div class="approval-info">
          <span class="approval-kind">${esc(kindLabels[a.kind]||a.kind)}</span>
          <div class="approval-summary">${esc(a.summary)}</div>
          <div class="approval-meta">${esc(a.topic_id||'')} · ${fmtRelative(a.proposed_at)} · estado: ${esc(a.status)}</div>
          ${a.units?`<details><summary class="muted">${a.units.length} unidades</summary>${a.units.map(u=>`<div class="muted" style="padding:4px 0">${u.order}. ${esc(u.concept_id)} — ${esc(u.reason)}</div>`).join('')}</details>`:''}
        </div>
        <div class="approval-actions">
          ${a.status==='pending'||a.status==='proposed'||a.status==='pending_approval'?`
            <button class="approve-btn" onclick="decideApproval('${esc(a.kind)}','${esc(a.id)}','approved')">Aprobar</button>
            <button class="reject-btn" onclick="decideApproval('${esc(a.kind)}','${esc(a.id)}','rejected')">Rechazar</button>`
          :`<span class="approval-status ${a.status}">${esc(a.status)}</span>`}
        </div>
      </div>`).join('')||'<div class="muted" style="text-align:center;padding:40px">Nada pendiente.</div>';
  }catch(e){el.innerHTML=`<div class="notice">${esc(e.message)}</div>`}
}

async function decideApproval(kind,id,decision){
  try{
    const endpoint={'memory_consolidation':'/api/agent/approvals/consolidation',
                    'mastery_inference':'/api/agent/approvals/mastery',
                    'roadmap':'/api/agent/approvals/roadmap',
                    'research_request':'/api/agent/approvals/research'}[kind];
    if(!endpoint){toast('Tipo de aprobación desconocido: '+kind);return}
    await api(endpoint,{method:'POST',body:JSON.stringify({id,decision:decision,decided_by:'web-user'})});
    toast(decision==='approved'?'Aprobado':'Rechazado');
    loadApprovals();
  }catch(e){toast(e.message)}
}

// ── Knowledge panel (documents + chunks browser) ─────────────────────────
async function loadKnowledge(){
  const el=$('#knowledge-documents');
  if(!el)return;
  try{
    const corpus=$('#knowledge-corpus')?.value||'reporter';
    const d=await api('/api/documents?corpus='+encodeURIComponent(corpus)+'&limit=100');
    const docs=d.documents||[];
    if(!docs.length){el.innerHTML='<div class="muted" style="text-align:center;padding:40px">Sin documentos en este corpus.</div>';return}
    el.innerHTML=docs.map(d=>`
      <div class="doc-item" onclick="viewDocument('${esc(d.path||'')}')">
        <strong>${esc(d.title||d.name||d.document_id||'')}</strong>
        <div class="muted">${esc(d.mime_type||'')} · ${fmtRelative(d.received_at||d.created_at||'')}</div>
      </div>`).join('');
  }catch(e){el.innerHTML=`<div class="notice">${esc(e.message)}</div>`}
}

// Process modal
let processTimer=null;
let currentProcess=null;
let logAutoFollow=true; // auto-scroll to bottom on new output

function showProcess(name){
  currentProcess=name;
  const detail=document.getElementById('process-detail');
  if(!detail)return;
  // Toggle: if clicking same process and visible, hide it
  if(detail.dataset.visible==='true' && detail.dataset.process===name){
    closeProcessModal();
    return;
  }
  // Show inline panel
  detail.dataset.visible='true';
  detail.dataset.process=name;
  detail.style.display='block';
  loadProcessDetail(name);
}

function closeProcessModal(){
  const detail=document.getElementById('process-detail');
  if(detail){
    detail.dataset.visible='false';
    detail.dataset.process='';
    detail.style.display='none';
    detail.innerHTML='';
  }
  currentProcess=null;
}

async function stopAllProcesses(){
  if(!confirm('¿Detener todos los procesos en ejecución? (scraper, fast path, reporter)'))return;
  try{
    const r=await fetch('/api/process/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(!r.ok){const e=await r.json().catch(()=>({error:r.statusText}));alert('Error: '+e.error);return}
    const data=await r.json();
    toast('Procesos detenidos: '+(data.stopped||[]).join(', '));
    refreshAll();
  }catch(e){alert('Error: '+e.message)}
}

async function cleanReporterCorpus(){
  if(!confirm('¿Limpiar el corpus del reporter? Se borrarán todos los índices (BM25, Tantivy, LanceDB), reportes y decisiones de curación. El corpus principal NO se toca.'))return;
  try{
    const r=await fetch('/api/corpus/clean',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(!r.ok){const e=await r.json().catch(()=>({error:r.statusText}));alert('Error: '+e.error);return}
    const data=await r.json();
    if(data.errors&&data.errors.length){alert('No se pudieron borrar (archivos bloqueados): '+data.errors.join(', '))}
    toast('Corpus reporter limpiado: '+(data.cleaned||[]).join(', '));
    selectedReport=null;
    refreshAll();
  }catch(e){alert('Error: '+e.message)}
}

async function startProcess(name,endpoint){
  try{
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(!r.ok){const e=await r.json().catch(()=>({error:r.statusText}));alert('Error: '+e.error);return}
    refreshAll();
  }catch(e){alert('Error: '+e.message)}
}

async function loadProcessDetail(name){
  const target=$('#process-detail');
  if(!target)return;
  // Save scroll positions before replacing content
  const logView=target.querySelector('.log-view');
  const logScrollTop=logView?logView.scrollTop:0;
  const logWasAtBottom=logView?(logView.scrollTop+logView.clientHeight>=logView.scrollHeight-5):true;
  // If user scrolled up manually, disable auto-follow
  if(logView&&!logWasAtBottom)logAutoFollow=false;
  try{
    const d=await api('/api/process?name='+encodeURIComponent(name));
    const metrics=d.metrics||{};
    const errors=d.errors||[];
    const log=(d.log||[]).map(l=>esc(l.replace(/\n$/,''))).join('\n');
    const metricsHtml=Object.keys(metrics).length?Object.entries(metrics).map(([k,v])=>`<div class="kv"><span class="muted">${esc(k)}</span><strong>${esc(String(v))}</strong></div>`).join(''):'<span class="muted">Sin métricas registradas.</span>';
    const errorsHtml=errors.length?`<div class="errors"><span class="eyebrow">ERRORES</span>${errors.map(e=>`<div class="error-item">${esc(String(e))}</div>`).join('')}</div>`:'';
    const pct=d.percent!=null?` · ${d.percent}%`:'';
    const stage=d.stage?` · ${esc(d.stage)}`:'';
    const isRunning=d.status==='running';
    const isError=d.status==='failed'||d.status==='error';
    const killBtn=(isRunning||isError)&&d.pid?`<button class="danger" onclick="killProcess(${d.pid},'${esc(name)}')">Matar proceso (PID ${d.pid})</button>`:'';
    const followBtn=`<button class="log-follow-btn ${logAutoFollow?'active':''}" onclick="toggleLogFollow()">${logAutoFollow?'● Siguiendo':'○ Seguir log'}</button>`;
    const procLabel=processLabels[name]||name;
    const logPath=d.log_path?`<div class="kv"><span class="muted">Log</span><strong style="font-size:11px;word-break:break-all">${esc(d.log_path)}</strong></div>`:'';
    target.innerHTML=`<div class="proc-header"><div><span class="eyebrow">PROCESO</span><h3>${esc(procLabel)}</h3></div><div><span class="${statusClass(d.status)}">${esc(d.status||'unknown')}${stage}${pct}</span> <button class="ghost small" onclick="closeProcessModal()">Cerrar</button></div></div>${killBtn?'<div class="proc-actions">'+killBtn+'</div>':''}<div class="proc-meta"><div class="kv"><span class="muted">PID</span><strong>${esc(d.pid??'—')}</strong></div><div class="kv"><span class="muted">Return code</span><strong>${esc(d.returncode??'—')}</strong></div><div class="kv"><span class="muted">Timestamp</span><strong>${fmtTime(d.timestamp)}</strong></div>${logPath}</div><div class="proc-section"><span class="eyebrow">MÉTRICAS</span><div class="metrics-grid">${metricsHtml}</div></div>${errorsHtml}<div class="proc-section"><span class="eyebrow">LOG (últimas ${d.log?d.log.length:0} líneas)</span>${followBtn}<pre class="log-view">${log||'(sin log disponible)'}</pre></div>`;
    // Restore scroll position after replacing content
    const newLogView=target.querySelector('.log-view');
    if(newLogView){
      if(logAutoFollow){
        // Auto-scroll to bottom (follow new output)
        newLogView.scrollTop=newLogView.scrollHeight;
      }else{
        // Preserve user's scroll position
        newLogView.scrollTop=logScrollTop;
      }
    }
    // Auto-refresh while process is running
    if(processTimer){clearInterval(processTimer);processTimer=null}
    if(isRunning){
      processTimer=setInterval(()=>{if(currentProcess===name)loadProcessDetail(name);else{clearInterval(processTimer);processTimer=null}},3000);
    }else{
      // Process finished: one final refresh to show the completed state.
      // The card in the overview updates via refreshAll, but the detail panel
      // was stuck showing the last "running" snapshot.
      if(currentProcess===name){
        setTimeout(()=>{if(currentProcess===name)loadProcessDetail(name)},1500);
      }
    }
  }catch(e){target.innerHTML=`<div class="notice">${esc(e.message)}</div>`}
}

function toggleLogFollow(){
  logAutoFollow=!logAutoFollow;
  const btn=document.querySelector('.log-follow-btn');
  if(btn){
    btn.textContent=logAutoFollow?'● Siguiendo':'○ Seguir log';
    btn.classList.toggle('active',logAutoFollow);
  }
  if(logAutoFollow){
    const logView=document.querySelector('.log-view');
    if(logView)logView.scrollTop=logView.scrollHeight;
  }
}

async function killProcess(pid,name){
  if(!confirm('¿Matar el proceso '+name+' (PID '+pid+')? Esto puede dejar archivos temporales.'))return;
  try{
    await api('/api/process/kill',{method:'POST',body:JSON.stringify({pid,name})});
    setTimeout(()=>loadProcessDetail(name),500);
  }catch(e){alert('No se pudo matar el proceso: '+e.message)}
}

function fmtTime(ts){
  if(!ts)return '—';
  const n=Number(ts);
  if(!isNaN(n)&&ts>1e9){const d=new Date(n*1000);return d.toLocaleString('es-AR')}
  return esc(String(ts));
}

setInterval(()=>{if(connectionState==='connected')refreshAll()},5000);

// Context menu for metric actions
let ctxMenu=null;
document.addEventListener('contextmenu',e=>{
  const metric=e.target.closest('.metric[data-ctx]');
  if(!metric)return;
  e.preventDefault();
  if(ctxMenu)ctxMenu.remove();
  const ctx=metric.dataset.ctx;
  const items=[];
  if(ctx==='landing'){
    items.push({label:'Limpiar Landing/web',desc:'Borra archivos scrapeados + índices DB + scrape_history.db',action:'cleanLanding'});
  }
  if(!items.length)return;
  ctxMenu=document.createElement('div');
  ctxMenu.className='ctx-menu';
  ctxMenu.innerHTML=items.map((it,i)=>`<div class="ctx-item" data-action="${it.action}"><strong>${esc(it.label)}</strong><br><small class="muted">${esc(it.desc)}</small></div>`).join('');
  document.body.appendChild(ctxMenu);
  const x=Math.min(e.clientX,window.innerWidth-280),y=Math.min(e.clientY,window.innerHeight-120);
  ctxMenu.style.left=x+'px';
  ctxMenu.style.top=y+'px';
  ctxMenu.querySelectorAll('.ctx-item').forEach(el=>{
    el.onclick=async()=>{
      const action=el.dataset.action;
      ctxMenu.remove();
      ctxMenu=null;
      if(action==='cleanLanding'){
        if(!confirm('¿Limpiar Landing/web? Se borrarán los archivos scrapeados, índices DB y scrape_history.db. El próximo scrape re-descargará todo.'))return;
        try{
          const res=await api('/api/landing/clean',{method:'POST',body:'{}'});
          toast(`Landing limpiada: ${res.cleaned?.length||0} elementos. Preservado: ${(res.preserved||[]).join(', ')}`);
          refreshAll();
        }catch(err){toast('Error: '+err.message)}
      }
    };
  });
});
document.addEventListener('click',()=>{if(ctxMenu){ctxMenu.remove();ctxMenu=null}});

// Initial connection
setConnectionState('connecting');
refreshAll();
