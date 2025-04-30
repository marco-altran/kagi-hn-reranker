from __future__ import annotations
import asyncio, os, random, signal, subprocess, time, textwrap, math
from typing import List
import aiohttp
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# ───────────────────────────── CONFIG ─────────────────────────────────────────
SKL_PORT = 8080
BGE_PORT = 8081

TOP_K        = 5
LLM_MODEL    = "gpt-4.1"
RANDOM_SEED  = 123

PROMPT_TMPL = textwrap.dedent("""\
    You are given a USER BIO and two unordered lists of Hacker News titles.
    Decide which list is overall **more relevant** to the bio.

    Respond with exactly one word:
    A   – List A is more relevant
    B   – List B is more relevant
    Tie – Both lists are equally relevant

    USER BIO:
    <<<{bio}>>>

    LIST A:
    {list_a}

    LIST B:
    {list_b}
""")

bios: List[str] = [
    # 1
    "I am an astrophysicist studying exoplanet atmospheres with Python, Pandas, and NASA’s ExoSims, and I follow advances in JWST instrumentation and open-source orbital mechanics libraries.",
    # 2
    "I’m a full-stack JavaScript engineer who loves serverless architectures on AWS, TypeScript, and edge-side rendering; outside of work I track WebGPU and Rust-based tooling.",
    # 3
    "I’m an environmental economist using Stata and R to model carbon pricing effects on emerging markets; keen on ESG reporting frameworks and green-bond innovations.",
    # 4
    "As a documentary filmmaker editing in DaVinci Resolve with Blackmagic RAW, I explore stories at the intersection of AI ethics and indigenous knowledge systems.",
    # 5
    "I’m a linguistics PhD focused on low-resource language parsing with spaCy and transformer finetuning; I monitor NLP fairness benchmarks and multilingual corpora releases.",
    # 6
    "I am a precision-agriculture entrepreneur deploying LoRaWAN sensor networks with Grafana dashboards and edge ML models in MicroPython to optimize irrigation.",
    # 7
    "I’m a quantitative trader coding in C++, KDB+/q, and Python, researching options market microstructure and GPU-accelerated Monte Carlo methods.",
    # 8
    "I’m a pediatrician turned digital-health product manager shipping Flutter apps, HL7/FHIR integrations, and studying privacy-preserving analytics for remote patient monitoring.",
    # 9
    "As a civic-tech activist I work with PostgreSQL/PostGIS and QGIS to map public-transport equity; interested in policy around open data mandates and micromobility.",
    # 10
    "I’m a cyber-security analyst running Zeek and Suricata on high-throughput NDR pipelines; I track zero-day exploitation trends and post-quantum cryptography.",
    # 11
    "I’m a marine biologist tagging sea turtles with Argos telemetry and processing movement data in R; passionate about conservation drones and habitat-mapping satellites.",
    # 12
    "I’m a DevOps engineer automating Kubernetes deployments with ArgoCD and Crossplane; following eBPF observability and WASM sidecar runtimes.",
    # 13
    "I am a music technologist building neural audio plugins in JUCE/C++; interested in differentiable DSP and open-source spatial audio toolkits.",
    # 14
    "I’m a humanitarian logistics coordinator using ODK, QGIS, and OpenStreetMap data to plan disaster-relief supply chains; exploring drone-based medical deliveries.",
    # 15
    "I’m a climate-risk actuary modeling catastrophe scenarios in R and Stan; following GHG-scenario alignment metrics and insurance-linked securities.",
    # 16
    "I’m a graphics programmer working with Vulkan, Rust, and RenderDoc to prototype path-traced real-time rendering; I follow GPU architecture leaks closely.",
    # 17
    "I’m a bioinformatician processing single-cell RNA-seq data using Bioconductor and Nextflow pipelines; fascinated by graph neural networks for gene-regulatory inference.",
    # 18
    "I’m a SaaS growth marketer building dashboards in Looker and dbt, A/B testing with Optimizely, and experimenting with causal inference in uplift modeling.",
    # 19
    "As a policy scholar I apply natural-language-processing in R to legislative corpora, studying polarization; I track GPT-4o civic-tech applications and data-privacy law.",
    # 20
    "I’m a sustainable-architecture consultant modeling passive-house envelopes in Rhino-Grasshopper with Ladybug; I track embodied-carbon standards and mass-timber innovations.",
    # 21
    "I’m an AR/VR developer building Unity XR apps with HDRP and OpenXR; watching Apple’s visionOS SDK and hand-tracking libraries.",
    # 22
    "I’m a satellite-operations engineer scripting in Go for ground-station automation, and tuning attitude-control algorithms in STK; keen on optical inter-satellite links.",
    # 23
    "I’m a behavioral economist running online experiments with oTree, analyzing in Python/PyMC3, and fascinated by reproducibility best practices.",
    # 24
    "I’m a fermentation scientist using LabVIEW and Raman spectroscopy to optimize yeast strains; following AI-driven strain-design startups.",
    # 25
    "I’m a fintech product designer prototyping flows in Figma, instrumenting Mixpanel, and researching CBDC wallet UX patterns.",
    # 26
    "I’m a volcanologist modeling ash dispersion with Fortran-based plume codes and visualizing outputs in ParaView; I track InSAR datasets and drone photogrammetry.",
    # 27
    "I’m a computational sociologist scraping Twitter APIs with Twarc and running network analyses in Gephi; interested in memetic contagion models.",
    # 28
    "I’m a quantum-computing researcher coding variational circuits in Qiskit and PennyLane, benchmarking on trapped-ion hardware; I follow error-mitigation papers.",
    # 29
    "I’m a wildlife photographer editing in Capture One, training YOLOv8 for species detection, and fascinated by low-light sensor tech advances.",
    # 30
    "I’m a legal-tech founder building GPT-powered contract summarizers with LangChain and Pinecone; I watch regulation around AI practice of law.",
    # 31
    "I’m an operations-research analyst solving vehicle-routing problems in Julia/JuMP and CPLEX; exploring reinforcement-learning dispatch heuristics.",
    # 32
    "I’m a computational chemist running DFT in ORCA, parsing results with ASE, and curious about graph networks for drug discovery.",
    # 33
    "I’m a humanitarian cartographer using RapiD and HOT Tasking Manager to update remote-area maps; researching ethics of AI-generated mapping.",
    # 34
    "I’m a renewable-energy project manager modeling PV output in SAM, tracking utility-scale storage LCOE, and learning about grid-forming inverters.",
    # 35
    "I’m a high-school CS teacher using Micro:bit and Python to teach embedded systems; following block-based coding research and pedagogy studies.",
    # 36
    "I’m a performance-marketing data engineer ETL-ing GA4 data with Airbyte, Warehousing on BigQuery; deep into Privacy Sandbox and server-side tagging.",
    # 37
    "I’m a humanitarian drone pilot processing imagery with Pix4D and QGIS to assist flood-mapping; researching BVLOS regulation.",
    # 38
    "I’m a sociolinguist analyzing TikTok sound patterns with Praat scripting; interested in algorithmic culture and youth dialect shifts.",
    # 39
    "I’m a robotics engineer programming ROS2 on Foxy, integrating LiDAR-SLAM with Nav2; watching NVIDIA Isaac and moveit2 developments.",
    # 40
    "I’m a CFO automating FP&A workflows in Anaplan, building Python forecasting notebooks, and following SEC ESG disclosure rules.",
    # 41
    "I’m an oceanographer deploying Seabird CTDs, cleaning data in MATLAB, and studying ENSO teleconnections with wavelet coherence.",
    # 42
    "I’m a PHP/Laravel backend developer optimizing MySQL, exploring Octane, and experimenting with Rust extensions via FFI.",
    # 43
    "I’m an AI safety researcher implementing red-teaming harnesses in Python, studying interpretability techniques and governance frameworks.",
    # 44
    "I’m a ceramic artist using 3D-printed clay molds, glazing with material-science insights, and interested in open-source kiln controllers.",
    # 45
    "I’m a social-impact VC analyst building discounted-cash-flow models in Excel, using Alphalens in Python for factor analysis, and tracking impact-measurement standards.",
    # 46
    "I’m a supply-chain strategist modeling port disruptions in AnyLogic, integrating SAP IBP, and diving into blockchain traceability pilots.",
    # 47
    "I’m an ed-tech founder deploying JupyterHub clusters on Kubernetes, crafting interactive STEM curricula and monitoring LTI 1.3 standards.",
    # 48
    "I’m a public-health epidemiologist running EpiModel in R and agent-based simulations in NetLogo; following wastewater surveillance advances.",
    # 49
    "I’m a 3D printing enthusiast designing in Fusion 360, slicing in Superslicer, and tracking open-source CoreXY builds.",
    # 50
    "I’m a computational journalist scraping FOIA datasets with Scrapy, visualizing in D3, and exploring LLM-assisted fact-checking.",
    # 51
    "I’m a neuroscientist recording hippocampal LFPs with Open Ephys, analyzing in MATLAB, and exploring Neuropixels 2.0 data pipelines.",
    # 52
    "I’m a sports-analytics developer building Python dashboards in Streamlit, ingesting StatsBomb JSON, and experimenting with tracking-data xG models.",
    # 53
    "I’m a security-oriented kernel engineer writing Rust for Linux drivers, exploring confidential computing and TDX, and fuzzing with Syzkaller.",
    # 54
    "I’m a renewable-hydrogen process engineer modelling electrolyzers in Aspen Plus, following green-ammonia value chains and energy markets.",
    # 55
    "I’m a poet using GPT-based co-writing tools, typesetting chapbooks in LaTeX, and researching blockchain-based royalty tracking.",
    # 56
    "I’m a digital-humanities scholar encoding TEI XML, running topic modeling in MALLET, and curious about AI-generated paleography.",
    # 57
    "I’m a game-economy designer balancing F2P systems in Python, simulating cohorts in R, and studying Web3 ownership dynamics.",
    # 58
    "I’m a carbon-accounting consultant automating GHG reports in Excel Power Query, using ecoinvent, and watching the EU CSRD rollout.",
    # 59
    "I’m a materials engineer characterizing solid-state batteries with XRD and COMSOL multiphysics; following dendrite-mitigation research.",
    # 60
    "I’m a wildfire risk analyst running FlamMap, processing MODIS data in Google Earth Engine, and studying controlled-burn policy.",
    # 61
    "I’m a biotech IP attorney annotating patents with Kira, drafting using GPT-assistants, and monitoring CRISPR licensing cases.",
    # 62
    "I’m a crowdsourcing researcher designing experiments on Prolific, analyzing with R tidyverse, and interested in large-scale annotation governance.",
    # 63
    "I’m a VR fitness startup founder writing Godot GDScript, integrating heart-rate APIs, and tracking OpenXR haptic standards.",
    # 64
    "I’m a computational finance professor teaching QuantLib in C++, exploring differentiable PDE solvers and federated learning in banking.",
    # 65
    "I’m an urban planner using CityEngine and ArcGIS Urban to model zoning scenarios, tracking 15-minute-city debates and bike-lane datasets.",
    # 66
    "I’m a backend Go developer leveraging gRPC, temporal.io workflows, and DynamoDB; experimenting with WebAssembly microservices.",
    # 67
    "I’m a nuclear engineer simulating reactor transients in RELAP5, performing uncertainty quantification in Python, and interested in SMR licensing.",
    # 68
    "I’m a digital forensics specialist parsing mobile device dumps with Cellebrite, scripting bulk hash matches in Python, and tracking on-device AI.",
    # 69
    "I’m a humanitarian water-sanitation engineer modeling groundwater flows in MODFLOW and designing solar pumping systems; interested in IoT-chlorination.",
    # 70
    "I’m a music-industry analyst scraping Spotify API with Next.js, clustering tracks in scikit-learn, and tracking AI voice cloning legislation.",
    # 71
    "I’m a climate activist building Nuxt3 JAMstack sites, managing Netlify functions, and tracking low-carbon web performance standards.",
    # 72
    "I’m a speech-pathologist studying dysarthria using Praat and automatic formant extraction in Python; following AI-based speech-therapy apps.",
    # 73
    "I’m a mechanical CAD engineer designing e-bikes in SolidWorks, running FEA in Ansys, and exploring open-source battery-management systems.",
    # 74
    "I’m an archival librarian digitizing microfilm with OCR pipelines, describing items in Dublin Core, and skeptical about AI auto-cataloguing.",
    # 75
    "I’m a planetary scientist using USGS ISIS3 and GDAL to build lunar DEMs, and excited about CLPS lander camera datasets.",
    # 76
    "I’m a chem-ed professor deploying PhET simulations, flipping classrooms with H5P, and assessing VR-based lab pedagogy.",
    # 77
    "I’m a biomedical engineer designing 3-D printed prosthetics in FreeCAD, controlling them with Arduino-based EMG sensors, and following bionic-hand research.",
    # 78
    "I’m a yacht-design naval architect running CFD in OpenFOAM, using Rhino plugin Orca3D, and curious about hydrogen-foil propulsion.",
    # 79
    "I’m a content strategist optimizing SEO in Ahrefs, producing data-rich narratives with Observable, and experimenting with AI-generated outlines.",
    # 80
    "I’m a humanitarian AI ethicist auditing facial-recognition datasets, building fairness dashboards in Streamlit, and lobbying for EU AI Act safeguards.",
    # 81
    "I’m a field ecologist deploying camera traps, automating identification with MegaDetector, and tracking citizen-science platforms.",
    # 82
    "I’m a microfluidics researcher simulating multiphase flow in COMSOL, fabricating chips with soft lithography, and integrating AI particle-tracking.",
    # 83
    "I’m a VMware admin automating vSphere with PowerCLI and Terraform, monitoring with Prometheus, and exploring homelab-scale Ceph.",
    # 84
    "I’m a space-policy analyst scraping FCC payload filings, visualizing orbits in CesiumJS, and studying debris-mitigation treaties.",
    # 85
    "I’m a cultural historian mining digitized newspapers with TopicRank, applying network visualizations in Gephi, and tracking digitization grants.",
    # 86
    "I’m a fashion-tech entrepreneur using CLO3D and Stoff.ai to create parametric garments, experimenting with supply-chain NFTs.",
    # 87
    "I’m a DAO governance researcher parsing Snapshot votes in Python, modeling token-based reputation, and tracking onchain quadratic funding.",
    # 88
    "I’m a psychology grad student running EEG with BrainVision, analyzing ERPs in MNE-Python, and reading about VR exposure therapy.",
    # 89
    "I’m a paleoclimatologist drilling ice cores, running isotope mass spectrometry on ICP-MS, and using Python xarray for paleodata analysis.",
    # 90
    "I’m a digital-fabrication instructor teaching laser-cutting with LightBurn, CNC with VCarve, and monitoring Fab Lab licensing news.",
    # 91
    "I’m an indie mobile dev building Flutter apps with Riverpod, integrating in-app purchases, and experimenting with Rive animations.",
    # 92
    "I’m a humanitarian crisis mapper processing SAR imagery in SNAP, using Google Earth Engine, and studying ethics of crowdsourced conflict data.",
    # 93
    "I’m a geotechnical engineer interpreting CPT/FEM scans in Plaxis, designing earth-retention, and tracking AI-aided soil classification.",
    # 94
    "I’m a computational philosopher building ontologies in Protegé, running logical inference in Hets, and examining AI epistemology.",
    # 95
    "I’m a poultry geneticist sequencing genomes with Illumina and assembling with SPAdes; intrigued by CRISPR gene drives.",
    # 96
    "I’m a Kubernetes SRE troubleshooting CNI plugins, writing Falco rules, and experimenting with Kube-Virt for VM workloads.",
    # 97
    "I’m a CRM architect customizing Salesforce Lightning with LWC, integrating MuleSoft, and learning about CDP real-time data.",
    # 98
    "I’m a theater lighting designer programming ETC Eos, simulating rigs in Capture, and curious about LED pixel-mapping software.",
    # 99
    "I’m a digital cartographer styling vector tiles in Mapbox GL JS, serving with Tippecanoe, and exploring real-time geofencing.",
    # 100
    "I’m a mental-health chatbot researcher fine-tuning Llama.cpp models on consumer CPUs with GGUF, and auditing conversational safety.",
]


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────── helper functions ────────────────────────────────
async def wait_until_ready(url: str, timeout: int = 30):
    """Ping the /rerank endpoint until we get any 200 response or timeout."""
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as s:
        while time.time() < deadline:
            try:
                async with s.post(url, json={"bio": "ping"}, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Service at {url} did not start within {timeout}s")

async def top_titles(session: aiohttp.ClientSession, url: str, bio: str, k: int) -> List[str]:
    """Fetch the top-k titles from a reranker service, retrying once on failure."""
    for attempt in range(2):
        try:
            async with session.post(url, json={"bio": bio}, timeout=None) as r:
                r.raise_for_status()
                ranked = await r.json()
            return [item["title"] for item in ranked[:k]]
        except Exception as e:
            if attempt == 0:
                # first failure: wait and retry once
                await asyncio.sleep(10)
            else:
                # second failure: silent fail, just log and return empty
                print(f"Error fetching titles from {url} for bio '{bio}': {e}")
                return []
    return []

def format_titles(titles: List[str]) -> str:
    """Format a list of titles for the prompt."""
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))

async def judge_pair(bio: str, a_titles: List[str], b_titles: List[str]) -> int:
    """Return  1 if A wins,  –1 if B wins,  0 if tie."""
    prompt = PROMPT_TMPL.format(
        bio=bio,
        list_a=format_titles(a_titles),
        list_b=format_titles(b_titles)
    )
    rsp = client.chat.completions.create(
        model=LLM_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens=3,
        temperature = 0,
    )
    ans = rsp.choices[0].message.content.strip().upper()
    if ans.startswith("A"):  return  1
    if ans.startswith("B"):  return -1
    return 0

def start_process(cmd: List[str], name: str, port: int) -> subprocess.Popen:
    """Start a server process with the specified port."""
    full_cmd = cmd + ["--port", str(port)]
    print(f"Starting {name}: {' '.join(full_cmd)}")
    return subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_process(proc: subprocess.Popen, name: str):
    """Stop a process gracefully."""
    print(f"Stopping {name}")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

# ─────────────────────────────── main loop ───────────────────────────────────
async def run_eval():
    """Run the evaluation pipeline."""

    wins = losses = ties = 0
    async with aiohttp.ClientSession() as session:
        for bio in tqdm_asyncio(bios, desc="Evaluating bios"):
            # fetch both top-5 title lists - updated URLs to point to /rerank
            skl_titles, bge_titles = await asyncio.gather(
                top_titles(session, f"http://localhost:{SKL_PORT}/rerank", bio, TOP_K),
                top_titles(session, f"http://localhost:{BGE_PORT}/rerank", bio, TOP_K),
            )

            # randomise mapping to List A / List B
            if random.random() < 0.5:
                list_a, list_b = bge_titles, skl_titles
                factor = 1  # +1 means BGE is List A
            else:
                list_a, list_b = skl_titles, bge_titles
                factor = -1  # BGE is List B

            outcome = await judge_pair(bio, list_a, list_b)
            if outcome == 1 * factor:
                wins += 1  # BGE wins
            elif outcome == -1 * factor:
                losses += 1  # sklearn wins
            else:
                ties += 1

    # ───── summary and significance (sign test) ─────
    n = wins + losses
    win_rate = wins / n if n else 0
    p_val = 2 * sum(
        math.comb(n, k) * (0.5**n)
        for k in range(0, wins+1)
    ) if wins <= n/2 else \
        2 * sum(
            math.comb(n, k) * (0.5**n)
            for k in range(wins, n+1)
        )

    print("\n────────── FINAL REPORT ──────────")
    print(f"Total bios judged   : {len(bios)}")
    print(f"Pairs (excl. ties)  : {n}")
    print(f"BGE wins            : {wins}")
    print(f"sklearn wins        : {losses}")
    print(f"Ties                : {ties}")
    print(f"BGE win-rate        : {win_rate:.2%}")
    print(f"Two-sided binomial p: {p_val:.4f}")
    print("──────────────────────────────────")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    asyncio.run(run_eval())