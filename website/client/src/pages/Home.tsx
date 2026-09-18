import { useState, type FormEvent } from "react";
import { toast } from "sonner";
import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronRight,
  CircleHelp,
  FileCheck2,
  Fingerprint,
  Gauge,
  Github,
  LockKeyhole,
  Menu,
  Play,
  ScanLine,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Upload,
  X,
  Zap,
} from "lucide-react";

// --- API Integration Types ---
interface PredictionResponse {
  customer_name: string;
  stability_score: number;
  stress_risk: string;
  recommendation: string;
  positive_signals: string[];
  risk_signals: string[];
  categories: { transaction_description: string; category: string }[];
}

const DEFAULT_TRANSACTIONS = [
  { name: "Maharashtra State Electricity", category: "Utilities", amount: "−₹1,240", date: "Sep 14", tone: "neutral" },
  { name: "Acme Studio Payroll", category: "Income", amount: "+₹42,800", date: "Sep 12", tone: "positive" },
  { name: "Metro One Card", category: "Transport", amount: "−₹860", date: "Sep 10", tone: "neutral" },
  { name: "FreshCart Groceries", category: "Essentials", amount: "−₹2,180", date: "Sep 08", tone: "neutral" },
];

const signalCards = [
  { icon: TrendingUp, label: "Income consistency", value: "92%", note: "stable for 4 months", tone: "green" },
  { icon: ShieldCheck, label: "Savings buffer", value: "3.2×", note: "monthly essentials covered", tone: "green" },
  { icon: TrendingDown, label: "Spending pressure", value: "−18%", note: "down from last month", tone: "lime" },
];

function LogoMark() {
  return (
    <span className="logo-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  );
}

function ScoreRing({ score = 78 }: { score?: number }) {
  const circumference = 2 * Math.PI * 47;
  const dashOffset = circumference - (score / 100) * circumference;
  return (
    <div className="score-ring" aria-label={`Stability score ${score} out of 100`}>
      <svg viewBox="0 0 112 112" role="img">
        <circle className="ring-track" cx="56" cy="56" r="47" />
        <circle className="ring-value" cx="56" cy="56" r="47" strokeDasharray={circumference} strokeDashoffset={dashOffset} />
      </svg>
      <div className="score-copy">
        <span className="score-number">{Math.round(score)}</span>
        <span className="score-label">/ 100</span>
      </div>
    </div>
  );
}

function FlowChart() {
  return (
    <div className="flow-chart" aria-label="Stability score trend over the last six months">
      <div className="chart-y-axis"><span>100</span><span>75</span><span>50</span><span>25</span></div>
      <svg viewBox="0 0 540 180" preserveAspectRatio="none" role="img">
        <defs>
          <linearGradient id="chart-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#b8ff3d" stopOpacity="0.28" />
            <stop offset="1" stopColor="#b8ff3d" stopOpacity="0" />
          </linearGradient>
          <filter id="chart-glow"><feGaussianBlur stdDeviation="4" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <path className="grid-line" d="M0 12H540M0 65H540M0 118H540M0 171H540" />
        <path className="chart-area" d="M0 142 C45 135, 56 151, 91 126 S145 125, 180 116 S230 108, 270 89 S316 114, 352 79 S406 92, 444 57 S505 68, 540 27 V180 H0 Z" />
        <path className="chart-line" d="M0 142 C45 135, 56 151, 91 126 S145 125, 180 116 S230 108, 270 89 S316 114, 352 79 S406 92, 444 57 S505 68, 540 27" filter="url(#chart-glow)" />
        <circle className="chart-dot" cx="540" cy="27" r="5" />
      </svg>
      <div className="chart-x-axis"><span>Apr</span><span>May</span><span>Jun</span><span>Jul</span><span>Aug</span><span>Sep</span></div>
    </div>
  );
}

export default function Home() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchPrediction = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "Arjun S.",
          transactions: [
            { date: '2023-01-01', desc: 'NEFT SALARY CREDIT', withdrawal: 0, deposit: 5000, balance: 5000 },
            { date: '2023-01-05', desc: 'UPI RENT PAYMENT', withdrawal: 500, deposit: 0, balance: 4500 },
            { date: '2023-02-01', desc: 'NEFT SALARY CREDIT', withdrawal: 0, deposit: 5000, balance: 9500 },
            { date: '2023-02-05', desc: 'UPI RENT PAYMENT', withdrawal: 500, deposit: 0, balance: 9000 },
          ]
        })
      });

      if (!response.ok) throw new Error("Failed to fetch prediction");
      const data = await response.json();
      setPrediction(data);
      toast.success("Stability profile updated via AI model.");
    } catch (error) {
      console.error(error);
      toast.error("Could not connect to ML API. Is the server running?");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!email.trim() || !email.includes("@")) {
      toast.error("Enter a valid email to join the early access list.");
      return;
    }
    setSubmitted(true);
    toast.success("You’re on the list. We’ll be in touch soon.");
  };

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    setMobileNavOpen(false);
  };

  return (
    <main className="site-shell">
      <div className="noise" aria-hidden="true" />
      <nav className="site-nav">
        <button className="brand-button" onClick={() => scrollTo("top")} aria-label="CreditBridge home">
          <LogoMark />
          <span className="brand-word">Credit<span>Bridge</span></span>
        </button>
        <div className={`nav-links ${mobileNavOpen ? "is-open" : ""}`}>
          <button onClick={() => scrollTo("how-it-works")}>How it works</button>
          <button onClick={() => scrollTo("signals")}>Signals</button>
          <button onClick={() => scrollTo("responsible-ai")}>Responsible AI</button>
          <button className="mobile-close" onClick={() => setMobileNavOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>
        <div className="nav-actions">
          <button className="nav-text-button" onClick={() => scrollTo("demo")}>View demo <ArrowUpRight size={15} /></button>
          <button className="nav-cta" onClick={() => scrollTo("early-access")}>Get early access <ArrowRight size={15} /></button>
        </div>
        <button className="mobile-menu" onClick={() => setMobileNavOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
      </nav>

      <section className="hero-section" id="top">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-copy">
          <div className="eyebrow"><span className="pulse-dot" /> Financial clarity, made visible</div>
          <h1>Credit,<br /><em>without the</em><br />credit history.</h1>
          <p className="hero-lede">CreditBridge turns everyday transaction behavior into a clear, explainable financial stability profile — so a thin file never has to tell the whole story.</p>
          <div className="hero-actions">
            <button className="primary-button" onClick={() => scrollTo("demo")}>Explore the signal <ArrowRight size={17} /></button>
            <button className="play-button" onClick={() => { scrollTo("how-it-works"); toast("A three-step view of the CreditBridge signal."); }}><span className="play-icon"><Play size={12} fill="currentColor" /></span> See how it works</button>
          </div>
          <div className="hero-meta"><LockKeyhole size={13} /> Encrypted by design <span className="meta-divider" /> <Fingerprint size={13} /> No traditional credit history required</div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="orb orb-one" />
          <div className="orb orb-two" />
          <div className="orb orb-three" />
          <div className="vertical-label">BRIDGE / SIGNAL 001</div>
          <div className="coordinate-label">19° 04′ 50.1″ N<br />72° 52′ 17.6″ E</div>
          <div className="hero-scan"><ScanLine size={16} /> LIVE MODEL / STABILITY INDEX</div>
        </div>
        <div className="hero-bottom-line"><span>READING FINANCIAL BEHAVIOR</span><span className="line-grow" /><span>01 / 04</span></div>
      </section>

      <section className="proof-strip">
        <div className="proof-intro"><span className="section-index">01</span><span>One signal,<br />many inputs.</span></div>
        <div className="proof-stat"><strong>0–100</strong><span>transparent score</span></div>
        <div className="proof-stat"><strong>7+</strong><span>behavioral signals</span></div>
        <div className="proof-stat"><strong>100%</strong><span>explainable by design</span></div>
        <div className="proof-note"><Sparkles size={17} /><span>Built for the<br /><strong>credit-invisible.</strong></span></div>
      </section>

      <section className="story-section" id="how-it-works">
        <div className="section-heading">
          <div><span className="kicker">THE MISSING CONTEXT</span><h2>Financial behavior<br /><em>leaves a trail.</em></h2></div>
          <p>We read the trail, not the label. CreditBridge translates patterns in your bank statement into the kind of context a traditional score was never designed to see.</p>
        </div>
        <div className="story-grid">
          <div className="story-card story-card-dark"><div className="card-topline"><span>01 / INGEST</span><Upload size={17} /></div><div className="file-chip"><FileCheck2 size={18} /><span>bank_statement.xlsx</span><span className="file-ready">READY</span></div><p>Securely clean and normalize your statement data.</p><div className="card-foot"><span>transaction rows</span><strong>1,284</strong></div></div>
          <div className="story-card story-card-light"><div className="card-topline"><span>02 / INTERPRET</span><BarChart3 size={17} /></div><div className="mini-bars"><span style={{ height: "42%" }} /><span style={{ height: "68%" }} /><span style={{ height: "55%" }} /><span style={{ height: "80%" }} /><span className="bar-hot" style={{ height: "94%" }} /><span style={{ height: "72%" }} /><span style={{ height: "88%" }} /></div><p>Find patterns in income, spending, buffers, and balance trends.</p><div className="card-foot"><span>signals found</span><strong>07</strong></div></div>
          <div className="story-card story-card-accent"><div className="card-topline"><span>03 / EXPLAIN</span><Gauge size={17} /></div><div className="score-preview"><ScoreRing score={prediction?.stability_score ?? 78} /><div><span className="preview-label">STABILITY SCORE</span><strong className="capitalize">{prediction?.recommendation.split(' ').slice(0, 2).join(' ') || "Strong footing"}</strong><small>{prediction?.risk_signals[0] || "+12 pts this month"}</small></div></div><p>See what’s working, what needs attention, and what to do next.</p><div className="card-foot"><span>decision support</span><strong>HUMAN-FIRST</strong></div></div>
        </div>
      </section>

      <section className="demo-section" id="demo">
        <div className="demo-heading"><div><span className="kicker">THE SIGNAL / LIVE PREVIEW</span><h2>Your finances,<br /><em>in focus.</em></h2></div><div className="demo-heading-copy"><span className="live-badge"><span className="pulse-dot" /> LIVE MODEL OUTPUT</span><p>Connecting directly to the behavioral AI pipeline.</p></div></div>
        <div className="dashboard-shell">
          <aside className="dashboard-rail"><div className="rail-brand"><LogoMark /><span>CB</span></div><div className="rail-menu"><span className="rail-item active"><Gauge size={17} /> <b>Overview</b></span><span className="rail-item"><BarChart3 size={17} /> Insights</span><span className="rail-item"><ShieldCheck size={17} /> Privacy</span></div><div className="rail-bottom"><span className="rail-avatar">{prediction?.customer_name?.slice(0,2).toUpperCase() || "AS"}</span><span>{prediction?.customer_name || "Arjun S."}</span><ChevronRight size={15} /></div></aside>
          <div className="dashboard-main"><div className="dashboard-top"><div><span className="dash-kicker">OVERVIEW / LIVE ANALYSIS</span><h3>Welcome, {prediction?.customer_name || "Arjun S."}</h3></div><button className="dashboard-action" onClick={fetchPrediction} disabled={isLoading}><Upload size={15} /> {isLoading ? "Analyzing..." : "Refresh Signal"}</button></div><div className="dashboard-metrics"><div className="score-panel"><div className="panel-heading"><span>FINANCIAL STABILITY</span><CircleHelp size={15} /></div><div className="score-content"><ScoreRing score={prediction?.stability_score ?? 78} /><div className="score-status"><span className={`status-pill ${prediction?.stress_risk === 'High' ? 'risk' : 'healthy'}`}>{prediction?.stress_risk || "HEALTHY"}</span><h4 className="capitalize">{prediction?.recommendation.split(' ').slice(0, 3).join(' ') || "Strong footing"}</h4><p>{prediction?.risk_signals[0] || "↑ 12 points from last month"}</p></div></div><div className="score-explainer"><div className="explainer-line"><span>Why this score?</span><ArrowUpRight size={14} /></div><p>{prediction?.recommendation || "Your income is consistent and your essential spending is well within your monthly inflow."}</p></div></div><div className="chart-panel"><div className="panel-heading"><span>STABILITY TREND</span><span className="trend-up"><ArrowUpRight size={14} /> 18.4%</span></div><FlowChart /><div className="chart-note"><span className="chart-legend" /> Score trend <span className="chart-date">Apr — Sep 2025</span></div></div></div><div className="dashboard-bottom"><div className="signals-panel"><div className="panel-heading"><span>KEY SIGNS</span><button onClick={() => scrollTo("signals")}>View all <ArrowRight size={14} /></button></div><div className="signal-list">{prediction?.positive_signals?.map((sig, i) => <div className="signal-row" key={i}><span className="signal-icon green"><Check size={16} /></span><div className="signal-name"><strong>Positive</strong><span>{sig}</span></div><strong className="signal-value">Stable</strong><ArrowRight size={15} className="signal-arrow" /></div>) || signalCards.map(({ icon: Icon, label, value, note, tone }) => <div className="signal-row" key={label}><span className={`signal-icon ${tone}`}><Icon size={16} /></span><div className="signal-name"><strong>{label}</strong><span>{note}</span></div><strong className="signal-value">{value}</strong><ArrowRight size={15} className="signal-arrow" /></div>)}</div></div><div className="activity-panel"><div className="panel-heading"><span>RECENT ACTIVITY</span><button onClick={() => toast("Full transaction history is available in the product.")}>See all <ArrowRight size={14} /></button></div><div className="transaction-list">{prediction?.categories?.map((tx, i) => <div className="transaction-row" key={i}><span className={`transaction-mark ${tx.category === 'Income' ? 'positive' : 'neutral'}`} /> <div className="transaction-name"><strong>{tx.transaction_description}</strong><span>{tx.category}</span></div><strong className={tx.category === 'Income' ? "amount-positive" : ""}>-</strong></div>) || DEFAULT_TRANSACTIONS.map((transaction) => <div className="transaction-row" key={transaction.name}><span className={`transaction-mark ${transaction.tone}`} /> <div className="transaction-name"><strong>{transaction.name}</strong><span>{transaction.category} · {transaction.date}</span></div><strong className={transaction.tone === "positive" ? "amount-positive" : ""}>{transaction.amount}</strong></div>)}</div></div></div></div>
        </div>
      </section>
      <section className="signals-section" id="signals">
        <div className="signals-aside"><span className="section-index">02</span><span className="kicker">BEYOND THE SCORE</span><h2>Context you<br /><em>can act on.</em></h2><p>A score is only useful when it comes with a reason. Every CreditBridge output is tied to a visible, human-readable signal.</p><button className="text-link" onClick={() => toast("Signal definitions are being prepared for the public beta.")}>Explore signal library <ArrowRight size={16} /></button></div>
        <div className="signal-tiles"><div className="large-signal-tile"><div className="tile-visual"><div className="tile-orbit orbit-a" /><div className="tile-orbit orbit-b" /><div className="tile-core">₹</div></div><div><span className="tile-number">01</span><h3>Cash-flow rhythm</h3><p>Understand how predictable your inflows and outflows feel month to month.</p></div><ArrowUpRight className="tile-arrow" /></div><div className="large-signal-tile tile-lime"><div className="tile-visual pulse-visual"><div className="pulse-wave wave-one" /><div className="pulse-wave wave-two" /><div className="pulse-wave wave-three" /></div><div><span className="tile-number">02</span><h3>Buffer strength</h3><p>See how much breathing room you have when life gets less predictable.</p></div><ArrowUpRight className="tile-arrow" /></div></div>
      </section>
      <section className="responsible-section" id="responsible-ai"><div className="responsible-art"><div className="radar"><div className="radar-ring r-one" /><div className="radar-ring r-two" /><div className="radar-ring r-three" /><div className="radar-cross cross-x" /><div className="radar-cross cross-y" /><div className="radar-sweep" /><div className="radar-point point-one" /><div className="radar-point point-two" /></div><span>MODEL / HUMAN<br />OVERSIGHT</span></div><div className="responsible-copy"><span className="section-index">03</span><span className="kicker">RESPONSIBLE BY DEFAULT</span><h2>A score should<br /><em>open doors.</em></h2><p>CreditBridge is a decision-support tool, not an automated gatekeeper. We show our work, protect your data, and keep a human in the loop before any consequential financial action.</p><div className="responsible-list"><div><Check size={16} /><span>Transparent feature weighting</span></div><div><Check size={16} /><span>No protected identity inference</span></div><div><Check size={16} /><span>Human review required</span></div></div></div></section>
      <section className="early-access-section" id="early-access"><div className="early-access-inner"><div><span className="kicker">THE NEXT BRIDGE</span><h2>Make your signal<br /><em>visible.</em></h2><p>We’re building a more inclusive way to understand financial readiness. Join the early access list.</p></div><form className="access-form" onSubmit={handleSubmit}>{submitted ? <div className="success-state"><span className="success-icon"><Check size={20} /></span><div><strong>You’re on the list.</strong><span>We’ll share the next signal when it’s ready.</span></div></div> : <><label htmlFor="email">YOUR EMAIL</label><div className="input-row"><input id="email" type="email" placeholder="you@example.com" value={email} onChange={(event) => setEmail(event.target.value)} /><button type="submit" aria-label="Join early access"><ArrowRight size={18} /></button></div><span className="form-note"><LockKeyhole size={12} /> No spam. Unsubscribe anytime.</span></>}</form></div></section>
      <footer className="site-footer"><div className="footer-brand"><LogoMark /><span className="brand-word">Credit<span>Bridge</span></span><p>Financial clarity for the credit-invisible.</p></div><div className="footer-links"><div><span>EXPLORE</span><button onClick={() => scrollTo("how-it-works")}>How it works</button><button onClick={() => scrollTo("signals")}>Signals</button></div><div><span>COMPANY</span><button onClick={() => scrollTo("about")}>About us</button><button onClick={() => scrollTo("research")}>Research</button></div><div><span>CONNECT</span><button onClick={() => toast("X / Twitter link coming soon.")}>X / Twitter <ArrowUpRight size={12} /></button><button onClick={() => toast("GitHub link coming coming soon.")}><Github size={12} /> GitHub <ArrowUpRight size={12} /></button></div></div><div className="footer-bottom"><span>© 2025 CreditBridge Labs</span><span>Built for better context <span className="footer-spark">✳</span></span><span>Prototype · Not a credit score</span></div></footer>
    </main>
  );
}

export { Home };
