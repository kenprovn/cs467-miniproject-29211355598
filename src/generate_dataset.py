"""
Dataset Generator for Scholarship and Student Grant Recommender (MP-C06)
Student: Nguyễn Minh Khanh (29211355598)
Course: CS467 Advanced Artificial Intelligence - CS467C

This script generates synthetic yet realistic datasets:
- data/items.csv: 40 diverse scholarships across domains, degree levels, and countries.
- data/ratings.csv: 15 students rating 16 scholarships each (240 ratings, 60% matrix sparsity).
"""

import os
import random
import pandas as pd
import numpy as np

# Set fixed random seeds for strict reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def get_scholarships_data():
    """Returns 40 synthetic scholarships with realistic metadata and eligibility text."""
    scholarships = [
        {
            "item_id": "SCH_01",
            "grant_name": "Global AI Undergraduate Summer Research Fellowship",
            "country": "USA",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Open to undergraduate computer science and software engineering students passionate about "
                "artificial intelligence, machine learning, and neural networks. Requires demonstrated interest in "
                "AI research and commitment to an 8-week summer exchange laboratory placement in the United States."
            ),
            "focus_tags": "AI, computer science, undergraduate, machine learning, summer exchange, research",
        },
        {
            "item_id": "SCH_02",
            "grant_name": "DAAD International CS & AI Summer Exchange Grant",
            "country": "Germany",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Available for undergraduate bachelor students in computer science, informatics, and computational science. "
                "Provides full funding for a 3-month summer exchange visiting research position in German university AI labs "
                "specializing in applied artificial intelligence and intelligent algorithms."
            ),
            "focus_tags": "computer science, AI, Germany, summer exchange, undergraduate, applied algorithms",
        },
        {
            "item_id": "SCH_03",
            "grant_name": "Turing Future Computing & AI Summer Scholar Award",
            "country": "UK",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Designed for high-achieving undergraduate students studying computer science, mathematics, or cognitive systems. "
                "Offers an intensive summer research fellowship focusing on artificial intelligence theory, machine learning models, "
                "and international collaborative research exchange."
            ),
            "focus_tags": "AI, computing, summer exchange, undergraduate, UK, machine learning research",
        },
        {
            "item_id": "SCH_04",
            "grant_name": "Asia-Pacific Summer Research Exchange in Computer Science",
            "country": "Singapore",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Supports undergraduate computer science and data engineering students seeking an immersive summer exchange. "
                "Recipients conduct hands-on laboratory research in robotics, computer vision, and foundation AI models under "
                "faculty mentorship."
            ),
            "focus_tags": "computer science, summer exchange, undergraduate, AI research, robotics, Singapore",
        },
        {
            "item_id": "SCH_05",
            "grant_name": "Canada AI & Machine Intelligence Undergraduate Research Grant",
            "country": "Canada",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Full-time undergraduate students enrolled in computer science, software engineering, or data science programs. "
                "Aimed at students seeking summer research experience in deep learning, generative artificial intelligence, "
                "and computational perception."
            ),
            "focus_tags": "Canada, undergraduate, artificial intelligence, deep learning, summer research, computer science",
        },
        {
            "item_id": "SCH_06",
            "grant_name": "Stanford AI Visionaries Graduate Fellowship",
            "country": "USA",
            "degree_level": "Master/PhD",
            "eligibility_text": (
                "Prestigious graduate fellowship for students enrolled in Master or PhD degrees in computer science. "
                "Focuses on cutting-edge research in foundation models, self-supervised learning, and computer vision systems."
            ),
            "focus_tags": "graduate, PhD, computer vision, foundation models, AI, computer science",
        },
        {
            "item_id": "SCH_07",
            "grant_name": "Swiss Federal Robotics and Intelligent Systems Grant",
            "country": "Switzerland",
            "degree_level": "Master",
            "eligibility_text": (
                "Master-level funding for students in robotics, electrical engineering, and computer science. "
                "Special emphasis on autonomous mobile robots, reinforcement learning, and embedded edge intelligence."
            ),
            "focus_tags": "robotics, intelligent systems, master, reinforcement learning, Switzerland, AI",
        },
        {
            "item_id": "SCH_08",
            "grant_name": "Tokyo Tech Advanced Computing & Data Science Fellowship",
            "country": "Japan",
            "degree_level": "Master/PhD",
            "eligibility_text": (
                "Doctoral and master candidates in computer science, computational mathematics, and data analytics. "
                "Supports research in high-performance computing, distributed training, and neural architecture optimization."
            ),
            "focus_tags": "Japan, high-performance computing, data science, graduate, computer science",
        },
        {
            "item_id": "SCH_09",
            "grant_name": "European Cybersecurity & Cloud Resilience Grant",
            "country": "Netherlands",
            "degree_level": "Master",
            "eligibility_text": (
                "Dedicated to graduate students specializing in network security, cryptography, distributed systems, "
                "and cloud infrastructure resilience. Requires bachelor degree in computer science or related field."
            ),
            "focus_tags": "cybersecurity, cloud computing, cryptography, master, network security",
        },
        {
            "item_id": "SCH_10",
            "grant_name": "Nordic Sustainable Energy & Smart Grid Scholarship",
            "country": "Sweden",
            "degree_level": "Master",
            "eligibility_text": (
                "Funding for master students in electrical engineering, renewable energy, and power systems. "
                "Focuses on smart grid integration, battery energy storage, and decarbonization technologies."
            ),
            "focus_tags": "renewable energy, smart grid, electrical engineering, Sweden, sustainability",
        },
        {
            "item_id": "SCH_11",
            "grant_name": "MIT-Broad Biomedical Engineering Fellowship",
            "country": "USA",
            "degree_level": "PhD",
            "eligibility_text": (
                "Supports doctoral students conducting interdisciplinary research at the intersection of bioengineering, "
                "computational genomics, and biomedical signal processing for clinical diagnostic devices."
            ),
            "focus_tags": "biomedical engineering, genomics, healthcare, PhD, computational biology",
        },
        {
            "item_id": "SCH_12",
            "grant_name": "Cambridge Health Sciences & Digital Medicine Grant",
            "country": "UK",
            "degree_level": "Master",
            "eligibility_text": (
                "Designed for students in medical sciences, digital health, and epidemiology. Focuses on translational health informatics, "
                "biomedical technology, and electronic health record analysis."
            ),
            "focus_tags": "digital medicine, healthcare informatics, Cambridge, UK, master, health sciences",
        },
        {
            "item_id": "SCH_13",
            "grant_name": "Global Public Health & Epidemiology Initiative Award",
            "country": "Switzerland",
            "degree_level": "Master",
            "eligibility_text": (
                "Targeted at international students studying global health, infectious disease modeling, and preventive healthcare. "
                "Requires relevant academic background in medicine, biology, or biostatistics."
            ),
            "focus_tags": "public health, epidemiology, biostatistics, global health, Switzerland",
        },
        {
            "item_id": "SCH_14",
            "grant_name": "Wharton International FinTech & Business Leadership Award",
            "country": "USA",
            "degree_level": "Master",
            "eligibility_text": (
                "Merit scholarship for MBA and Master of Science in Finance students. Explores financial technology innovations, "
                "algorithmic trading, decentralized finance, and corporate venture capital."
            ),
            "focus_tags": "fintech, business leadership, MBA, finance, blockchain, USA",
        },
        {
            "item_id": "SCH_15",
            "grant_name": "London Business School Global Enterprise Fellowship",
            "country": "UK",
            "degree_level": "Master",
            "eligibility_text": (
                "For enterprising graduate students pursuing master degrees in management, international business, and entrepreneurship. "
                "Supports venture creation and emerging market economic strategy."
            ),
            "focus_tags": "business, management, entrepreneurship, global enterprise, UK, master",
        },
        {
            "item_id": "SCH_16",
            "grant_name": "INSEAD Sustainable Business & ESG Strategy Scholarship",
            "country": "France",
            "degree_level": "Master",
            "eligibility_text": (
                "Designed for business students researching environmental, social, and corporate governance (ESG) standards, "
                "sustainable supply chains, and green corporate transition leadership."
            ),
            "focus_tags": "ESG, sustainable business, corporate governance, France, MBA, sustainability",
        },
        {
            "item_id": "SCH_17",
            "grant_name": "Australian Clean Oceans & Marine Biology Research Grant",
            "country": "Australia",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Undergraduate research grant providing hands-on fieldwork in marine biology, oceanography, coral reef conservation, "
                "and coastal ecological protection in the Great Barrier Reef region."
            ),
            "focus_tags": "marine biology, ocean conservation, Australia, ecology, undergraduate research",
        },
        {
            "item_id": "SCH_18",
            "grant_name": "Kyoto Environmental Policy & Climate Adaptation Award",
            "country": "Japan",
            "degree_level": "Master",
            "eligibility_text": (
                "Interdisciplinary grant for master scholars studying international climate treaties, carbon offset mechanisms, "
                "and urban climate resilience planning across East Asia."
            ),
            "focus_tags": "environmental policy, climate change, adaptation, urban resilience, Japan",
        },
        {
            "item_id": "SCH_19",
            "grant_name": "Munich Automotive AI & Autonomous Driving Grant",
            "country": "Germany",
            "degree_level": "Master",
            "eligibility_text": (
                "Aimed at graduate engineering students working on automated driving, LiDAR sensor fusion, camera perception, "
                "and deep learning algorithms for vehicle safety."
            ),
            "focus_tags": "autonomous vehicles, automotive AI, sensor fusion, computer vision, Germany",
        },
        {
            "item_id": "SCH_20",
            "grant_name": "Oxford Humanities & Digital Ethics Research Fellowship",
            "country": "UK",
            "degree_level": "PhD",
            "eligibility_text": (
                "Doctoral fellowship investigating the philosophical, moral, and sociological implications of artificial intelligence, "
                "algorithmic bias, digital surveillance, and human rights in tech."
            ),
            "focus_tags": "digital ethics, humanities, AI governance, philosophy of technology, UK, PhD",
        },
        {
            "item_id": "SCH_21",
            "grant_name": "Harvard Social Justice & International Human Rights Grant",
            "country": "USA",
            "degree_level": "Master",
            "eligibility_text": (
                "For graduate students in international law, human rights advocacy, and public interest policy. "
                "Supports research on global inequality, migrant protection, and institutional reform."
            ),
            "focus_tags": "human rights, social justice, public policy, law, USA, master",
        },
        {
            "item_id": "SCH_22",
            "grant_name": "Seoul Smart Cities & IoT Infrastructure Award",
            "country": "South Korea",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Undergraduate grant for students in electrical engineering, computer science, or civil engineering developing smart "
                "city sensor architectures, wireless IoT networks, and intelligent traffic control."
            ),
            "focus_tags": "smart cities, IoT, urban infrastructure, South Korea, undergraduate",
        },
        {
            "item_id": "SCH_23",
            "grant_name": "Waterloo Quantum Computing & Information Scholarship",
            "country": "Canada",
            "degree_level": "PhD",
            "eligibility_text": (
                "Doctoral scholarship for students researching quantum algorithms, quantum error correction, superconducting qubits, "
                "and theoretical computer science at the Institute for Quantum Computing."
            ),
            "focus_tags": "quantum computing, quantum algorithms, theoretical CS, physics, Canada, PhD",
        },
        {
            "item_id": "SCH_24",
            "grant_name": "Sorbonne European Literature and Cultural Studies Grant",
            "country": "France",
            "degree_level": "Master",
            "eligibility_text": (
                "Scholarship for master students conducting scholarly research in comparative literature, cultural linguistics, "
                "European history, and literary translation across European languages."
            ),
            "focus_tags": "literature, cultural studies, humanities, linguistics, France, master",
        },
        {
            "item_id": "SCH_25",
            "grant_name": "Melbourne Biomedical Informatics & Genomics Fellowship",
            "country": "Australia",
            "degree_level": "PhD",
            "eligibility_text": (
                "For doctoral students pursuing translational genomics, single-cell RNA sequencing data analysis, "
                "and computational bioinformatics to enhance precision cancer therapeutics."
            ),
            "focus_tags": "bioinformatics, genomics, precision medicine, computational biology, Australia, PhD",
        },
        {
            "item_id": "SCH_26",
            "grant_name": "Berkeley Open Source Software & Systems Grant",
            "country": "USA",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Awarded to talented undergraduate computer science developers contributing to open-source operating systems, "
                "programming language compilers, cloud runtimes, and distributed software systems."
            ),
            "focus_tags": "open source, systems software, computer science, compilers, undergraduate, USA",
        },
        {
            "item_id": "SCH_27",
            "grant_name": "ETH Zurich Applied Mathematics & Cryptography Award",
            "country": "Switzerland",
            "degree_level": "Master/PhD",
            "eligibility_text": (
                "Funding for graduate students specializing in post-quantum cryptography, discrete mathematics, number theory, "
                "and formal mathematical verification of security protocols."
            ),
            "focus_tags": "cryptography, applied mathematics, cybersecurity, algorithms, Switzerland",
        },
        {
            "item_id": "SCH_28",
            "grant_name": "Singapore Global Logistics & Supply Chain Fellowship",
            "country": "Singapore",
            "degree_level": "Master",
            "eligibility_text": (
                "Master-level fellowship focusing on operations research, maritime freight optimization, automated warehousing, "
                "and intelligent supply chain analytics in trade hub networks."
            ),
            "focus_tags": "logistics, supply chain, operations research, industrial engineering, Singapore",
        },
        {
            "item_id": "SCH_29",
            "grant_name": "Max Planck Quantum Physics & Nanotechnology Fellowship",
            "country": "Germany",
            "degree_level": "PhD",
            "eligibility_text": (
                "Prestigious doctoral grant supporting experimental and theoretical research in solid-state quantum physics, "
                "nanophotonics, 2D materials, and atomic scale device fabrication."
            ),
            "focus_tags": "nanotechnology, quantum physics, materials science, Germany, PhD",
        },
        {
            "item_id": "SCH_30",
            "grant_name": "Edinburgh Natural Language Processing & Speech Award",
            "country": "UK",
            "degree_level": "Master",
            "eligibility_text": (
                "Dedicated to graduate students pursuing master degrees in computer science or linguistics specializing in NLP, "
                "conversational dialogue agents, automated machine translation, and speech synthesis."
            ),
            "focus_tags": "NLP, natural language processing, speech technology, linguistics, AI, UK, master",
        },
        {
            "item_id": "SCH_31",
            "grant_name": "Sydney Agritech & Sustainable Agriculture Grant",
            "country": "Australia",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Undergraduate scholarship for students in agricultural science, biological engineering, and precision farming "
                "utilizing satellite imagery, soil sensors, and sustainable crop genetics."
            ),
            "focus_tags": "agritech, sustainable agriculture, biotechnology, Australia, undergraduate",
        },
        {
            "item_id": "SCH_32",
            "grant_name": "Toronto AI in Medicine & Clinical Imaging Award",
            "country": "Canada",
            "degree_level": "PhD",
            "eligibility_text": (
                "Doctoral research funding for researchers developing convolutional neural networks, multi-modal medical imaging, "
                "and diagnostic AI tools for radiology and oncology clinical workflows."
            ),
            "focus_tags": "medical imaging, AI, deep learning, radiology, healthcare, Canada, PhD",
        },
        {
            "item_id": "SCH_33",
            "grant_name": "Stockholm Creative Arts & Interactive Media Fellowship",
            "country": "Sweden",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Undergraduate fellowship for creative digital artists, game designers, and UI/UX practitioners developing "
                "interactive virtual environments, digital storytelling, and human-computer creative interfaces."
            ),
            "focus_tags": "digital arts, game design, interactive media, UI/UX, Sweden, undergraduate",
        },
        {
            "item_id": "SCH_34",
            "grant_name": "Helsinki Space Technology & Satellite Communications Grant",
            "country": "Finland",
            "degree_level": "Master",
            "eligibility_text": (
                "Supports master students in aerospace engineering, radio technology, and space physics designing CubeSats, "
                "orbital communication transceivers, and satellite mission payloads."
            ),
            "focus_tags": "space technology, aerospace engineering, satellites, communications, Finland",
        },
        {
            "item_id": "SCH_35",
            "grant_name": "Columbia International Journalism & Media Integrity Grant",
            "country": "USA",
            "degree_level": "Master",
            "eligibility_text": (
                "For investigative journalism master students dedicated to uncovering public corruption, corporate accountability, "
                "disinformation campaigns, and freedom of the press globally."
            ),
            "focus_tags": "journalism, media integrity, public communication, USA, master",
        },
        {
            "item_id": "SCH_36",
            "grant_name": "Dublin Data Engineering & Distributed Systems Fellowship",
            "country": "Ireland",
            "degree_level": "Master",
            "eligibility_text": (
                "Master program funding for computer science students building streaming data architectures, Kafka pipelines, "
                "large-scale distributed databases, and fault-tolerant cloud processing engines."
            ),
            "focus_tags": "data engineering, distributed systems, cloud architectures, computer science, Ireland",
        },
        {
            "item_id": "SCH_37",
            "grant_name": "Milan Industrial Design & Sustainable Materials Award",
            "country": "Italy",
            "degree_level": "Master",
            "eligibility_text": (
                "For graduate product designers integrating circular economy concepts, biodegradable composites, and ergonomic "
                "principles into consumer electronics and sustainable industrial products."
            ),
            "focus_tags": "industrial design, sustainable materials, product design, Italy, master",
        },
        {
            "item_id": "SCH_38",
            "grant_name": "Munich Aerospace Propulsion & Aerodynamics Grant",
            "country": "Germany",
            "degree_level": "PhD",
            "eligibility_text": (
                "Doctoral fellowship for mechanical and aerospace researchers investigating supersonic fluid mechanics, "
                "hydrogen combustion propulsion, and lightweight composite aircraft structures."
            ),
            "focus_tags": "aerospace engineering, propulsion, aerodynamics, mechanical, Germany, PhD",
        },
        {
            "item_id": "SCH_39",
            "grant_name": "Imperial College Applied Robotics & Autonomous Drones Award",
            "country": "UK",
            "degree_level": "Master",
            "eligibility_text": (
                "For master students in engineering and computing developing unmanned aerial vehicles, real-time path planning, "
                "visual inertial odometry, and obstacle avoidance in complex environments."
            ),
            "focus_tags": "robotics, autonomous drones, navigation, computer vision, UK, master",
        },
        {
            "item_id": "SCH_40",
            "grant_name": "Global Undergraduate Summer STEM Exchange Program",
            "country": "USA",
            "degree_level": "Undergraduate",
            "eligibility_text": (
                "Broad summer exchange laboratory program for international undergraduate students in STEM fields (computer science, "
                "engineering, biotechnology, physics) seeking summer research mentorship and global exposure."
            ),
            "focus_tags": "STEM, summer exchange, undergraduate, research, computer science, engineering, USA",
        },
    ]
    return scholarships


def get_user_profiles():
    """Defines 15 student profiles covering varied fields, degree levels, and interests."""
    profiles = {
        "U01": {
            "name": "Undergrad CS & AI Student",
            "major": "Computer Science",
            "degree": "Undergraduate",
            "core_interests": ["AI", "computer science", "undergraduate", "summer exchange", "machine learning"],
            "positive_items": ["SCH_01", "SCH_02", "SCH_03", "SCH_04", "SCH_05", "SCH_26", "SCH_40"],
            "neutral_items": ["SCH_06", "SCH_07", "SCH_08", "SCH_09", "SCH_30", "SCH_36", "SCH_22", "SCH_19"],
            "negative_items": ["SCH_14", "SCH_15", "SCH_17", "SCH_24", "SCH_35", "SCH_37"],
        },
        "U02": {
            "name": "Undergrad Software Systems Student",
            "major": "Computer Science & Systems",
            "degree": "Undergraduate",
            "core_interests": ["computer science", "systems software", "open source", "distributed systems", "undergraduate"],
            "positive_items": ["SCH_26", "SCH_01", "SCH_02", "SCH_04", "SCH_36", "SCH_40", "SCH_05"],
            "neutral_items": ["SCH_03", "SCH_08", "SCH_09", "SCH_22", "SCH_27", "SCH_07"],
            "negative_items": ["SCH_11", "SCH_13", "SCH_16", "SCH_21", "SCH_24", "SCH_33"],
        },
        "U03": {
            "name": "Graduate AI & Robotics Student",
            "major": "Artificial Intelligence",
            "degree": "Master",
            "core_interests": ["AI", "robotics", "intelligent systems", "computer vision", "reinforcement learning"],
            "positive_items": ["SCH_06", "SCH_07", "SCH_19", "SCH_39", "SCH_01", "SCH_05", "SCH_08"],
            "neutral_items": ["SCH_02", "SCH_03", "SCH_04", "SCH_30", "SCH_32", "SCH_22"],
            "negative_items": ["SCH_15", "SCH_17", "SCH_21", "SCH_24", "SCH_31", "SCH_35"],
        },
        "U04": {
            "name": "Master Cybersecurity & Cloud Student",
            "major": "Information Security",
            "degree": "Master",
            "core_interests": ["cybersecurity", "cryptography", "cloud computing", "distributed systems"],
            "positive_items": ["SCH_09", "SCH_27", "SCH_36", "SCH_26", "SCH_08", "SCH_02"],
            "neutral_items": ["SCH_01", "SCH_03", "SCH_07", "SCH_22", "SCH_23"],
            "negative_items": ["SCH_12", "SCH_13", "SCH_17", "SCH_31", "SCH_33", "SCH_37"],
        },
        "U05": {
            "name": "Master FinTech & Management Student",
            "major": "Finance & Business",
            "degree": "Master",
            "core_interests": ["fintech", "finance", "business", "MBA", "management"],
            "positive_items": ["SCH_14", "SCH_15", "SCH_16", "SCH_28", "SCH_09"],
            "neutral_items": ["SCH_08", "SCH_20", "SCH_21", "SCH_35", "SCH_36"],
            "negative_items": ["SCH_01", "SCH_11", "SCH_17", "SCH_29", "SCH_38", "SCH_39"],
        },
        "U06": {
            "name": "PhD Biomedical Informatics Student",
            "major": "Bioinformatics & Health",
            "degree": "PhD",
            "core_interests": ["biomedical engineering", "genomics", "bioinformatics", "healthcare", "clinical imaging"],
            "positive_items": ["SCH_11", "SCH_12", "SCH_25", "SCH_32", "SCH_13", "SCH_06"],
            "neutral_items": ["SCH_08", "SCH_27", "SCH_30", "SCH_10", "SCH_01"],
            "negative_items": ["SCH_14", "SCH_15", "SCH_24", "SCH_28", "SCH_37", "SCH_38"],
        },
        "U07": {
            "name": "Undergrad Marine Biology & Ecology Student",
            "major": "Environmental Science",
            "degree": "Undergraduate",
            "core_interests": ["marine biology", "ocean conservation", "ecology", "sustainability"],
            "positive_items": ["SCH_17", "SCH_18", "SCH_10", "SCH_31", "SCH_40", "SCH_16"],
            "neutral_items": ["SCH_13", "SCH_21", "SCH_22", "SCH_33"],
            "negative_items": ["SCH_01", "SCH_06", "SCH_09", "SCH_23", "SCH_27", "SCH_36"],
        },
        "U08": {
            "name": "Undergrad AI Research & Exchange Candidate",
            "major": "Computer Science",
            "degree": "Undergraduate",
            "core_interests": ["undergraduate", "computer science", "AI research", "summer exchange", "machine learning"],
            "positive_items": ["SCH_01", "SCH_02", "SCH_03", "SCH_04", "SCH_05", "SCH_40", "SCH_26"],
            "neutral_items": ["SCH_06", "SCH_07", "SCH_08", "SCH_30", "SCH_22", "SCH_36"],
            "negative_items": ["SCH_11", "SCH_14", "SCH_15", "SCH_17", "SCH_24", "SCH_37"],
        },
        "U09": {
            "name": "Graduate Autonomous Systems Engineer",
            "major": "Mechanical & Robotics",
            "degree": "Master",
            "core_interests": ["autonomous vehicles", "robotics", "automotive AI", "propulsion", "drones"],
            "positive_items": ["SCH_19", "SCH_07", "SCH_39", "SCH_38", "SCH_08", "SCH_22"],
            "neutral_items": ["SCH_01", "SCH_04", "SCH_06", "SCH_10", "SCH_34"],
            "negative_items": ["SCH_12", "SCH_14", "SCH_15", "SCH_20", "SCH_24", "SCH_35"],
        },
        "U10": {
            "name": "PhD Quantum Computing & Physics Scholar",
            "major": "Quantum Information",
            "degree": "PhD",
            "core_interests": ["quantum computing", "quantum physics", "nanotechnology", "cryptography"],
            "positive_items": ["SCH_23", "SCH_29", "SCH_27", "SCH_08", "SCH_06"],
            "neutral_items": ["SCH_01", "SCH_03", "SCH_09", "SCH_34", "SCH_36"],
            "negative_items": ["SCH_14", "SCH_15", "SCH_17", "SCH_21", "SCH_31", "SCH_33"],
        },
        "U11": {
            "name": "Master Environmental Sustainability Student",
            "major": "Sustainable Development",
            "degree": "Master",
            "core_interests": ["sustainability", "renewable energy", "climate adaptation", "ESG"],
            "positive_items": ["SCH_10", "SCH_16", "SCH_18", "SCH_31", "SCH_17", "SCH_13"],
            "neutral_items": ["SCH_15", "SCH_21", "SCH_22", "SCH_28", "SCH_40"],
            "negative_items": ["SCH_01", "SCH_09", "SCH_23", "SCH_26", "SCH_29", "SCH_36"],
        },
        "U12": {
            "name": "Master NLP & Computational Linguistics Student",
            "major": "Computer Science / NLP",
            "degree": "Master",
            "core_interests": ["NLP", "speech technology", "computational linguistics", "AI ethics", "foundation models"],
            "positive_items": ["SCH_30", "SCH_20", "SCH_06", "SCH_01", "SCH_03", "SCH_05"],
            "neutral_items": ["SCH_02", "SCH_04", "SCH_08", "SCH_24", "SCH_35", "SCH_36"],
            "negative_items": ["SCH_10", "SCH_14", "SCH_17", "SCH_28", "SCH_31", "SCH_38"],
        },
        "U13": {
            "name": "Master Public Health & Digital Health Student",
            "major": "Public Health",
            "degree": "Master",
            "core_interests": ["public health", "epidemiology", "digital medicine", "healthcare informatics"],
            "positive_items": ["SCH_13", "SCH_12", "SCH_11", "SCH_25", "SCH_32"],
            "neutral_items": ["SCH_16", "SCH_18", "SCH_20", "SCH_21", "SCH_40"],
            "negative_items": ["SCH_01", "SCH_04", "SCH_09", "SCH_26", "SCH_36", "SCH_38"],
        },
        "U14": {
            "name": "Master International Business & Operations Student",
            "major": "Logistics & Global Business",
            "degree": "Master",
            "core_interests": ["logistics", "supply chain", "business enterprise", "management", "fintech"],
            "positive_items": ["SCH_28", "SCH_15", "SCH_14", "SCH_16", "SCH_09"],
            "neutral_items": ["SCH_10", "SCH_18", "SCH_21", "SCH_22", "SCH_36"],
            "negative_items": ["SCH_01", "SCH_03", "SCH_11", "SCH_23", "SCH_29", "SCH_32"],
        },
        "U15": {
            "name": "Undergrad Interactive Media & Design Student",
            "major": "Digital Media & HCI",
            "degree": "Undergraduate",
            "core_interests": ["interactive media", "digital arts", "UI/UX", "design", "humanities"],
            "positive_items": ["SCH_33", "SCH_37", "SCH_24", "SCH_20", "SCH_40", "SCH_26"],
            "neutral_items": ["SCH_01", "SCH_04", "SCH_15", "SCH_22", "SCH_35"],
            "negative_items": ["SCH_08", "SCH_11", "SCH_14", "SCH_23", "SCH_27", "SCH_29"],
        },
    }
    return profiles


def generate_ratings(items_df, profiles):
    """
    Generates ratings for 15 users rating exactly 16 scholarships each (240 ratings total).
    Matrix: 15 users x 40 items = 600 entries. Missing = 360 entries -> Sparsity = 60.0%.
    """
    all_items = items_df["item_id"].tolist()
    ratings_records = []

    # Comment templates to produce natural English notes
    high_notes = [
        "Highly aligned with my curriculum, career ambitions, and academic background.",
        "Excellent match for my research interests; funding and lab facilities are outstanding.",
        "Ideal opportunity for my degree level and domain specialization. Top recommendation.",
        "Matches my exact profile for research exchange and hands-on laboratory experience.",
        "Strong institutional reputation and exceptional synergy with my current projects.",
    ]
    mid_notes = [
        "Relevant technical domain, though partial mismatch in degree level or prerequisite scope.",
        "Interesting opportunity with solid potential, though slightly outside my core specialty.",
        "Good program overall, but requires additional coursework to qualify competitively.",
        "Decent alignment with my background, worth considering as a secondary alternative.",
    ]
    low_notes = [
        "Far outside my academic major and immediate research priorities.",
        "Eligibility criteria do not fit my current academic status or degree level.",
        "Minimal relevance to my field of study; unlikely to submit an application.",
        "Not aligned with my career trajectory or domain interests.",
    ]

    for user_id, profile in sorted(profiles.items()):
        pos = set(profile["positive_items"])
        neut = set(profile["neutral_items"])
        neg = set(profile["negative_items"])

        # We must select exactly 16 items for this user:
        # e.g., 5-6 positive, 5-6 neutral, 4-5 negative
        chosen_items = []
        available_pos = [item for item in pos if item in all_items]
        available_neut = [item for item in neut if item in all_items]
        available_neg = [item for item in neg if item in all_items]
        remaining = [item for item in all_items if item not in pos and item not in neut and item not in neg]

        # Sample with deterministic seeds per user
        user_num = int(user_id[1:])
        user_rng = random.Random(RANDOM_SEED + user_num * 10)

        n_pos = min(len(available_pos), user_rng.randint(5, 6))
        n_neut = min(len(available_neut), user_rng.randint(5, 6))
        n_neg = 16 - (n_pos + n_neut)

        selected_pos = user_rng.sample(available_pos, n_pos)
        selected_neut = user_rng.sample(available_neut, n_neut)
        
        pool_neg = available_neg + remaining
        pool_neg = [x for x in pool_neg if x not in selected_pos and x not in selected_neut]
        selected_neg = user_rng.sample(pool_neg, min(len(pool_neg), n_neg))

        user_items = selected_pos + selected_neut + selected_neg
        if len(user_items) < 16:
            rest = [x for x in all_items if x not in user_items]
            user_items.extend(user_rng.sample(rest, 16 - len(user_items)))
        elif len(user_items) > 16:
            user_items = user_items[:16]

        assert len(user_items) == 16, f"User {user_id} does not have exactly 16 items"

        for item_id in user_items:
            if item_id in pos:
                rating = user_rng.choice([4, 5])
                note = user_rng.choice(high_notes)
            elif item_id in neut:
                rating = 3
                note = user_rng.choice(mid_notes)
            else:
                rating = user_rng.choice([1, 2])
                note = user_rng.choice(low_notes)

            ratings_records.append({
                "user_id": user_id,
                "item_id": item_id,
                "relevance_rating": rating,
                "student_note": note,
            })

    ratings_df = pd.DataFrame(ratings_records)
    return ratings_df


def validate_dataset(items_df, ratings_df):
    """Performs rigorous automated checks on schema, integrity, counts, and sparsity."""
    print("=" * 60)
    print("DATASET VALIDATION CHECKS")
    print("=" * 60)

    # 1. Check items schema
    expected_item_cols = ["item_id", "grant_name", "country", "degree_level", "eligibility_text", "focus_tags"]
    assert list(items_df.columns) == expected_item_cols, f"Invalid items columns: {items_df.columns}"
    assert len(items_df) == 40, f"Expected 40 items, got {len(items_df)}"
    assert items_df.isnull().sum().sum() == 0, "Missing values detected in items.csv"
    print(f"✓ items.csv schema verified: {len(items_df)} scholarships, 0 nulls.")

    # 2. Check ratings schema
    expected_rating_cols = ["user_id", "item_id", "relevance_rating", "student_note"]
    assert list(ratings_df.columns) == expected_rating_cols, f"Invalid ratings columns: {ratings_df.columns}"
    assert len(ratings_df) == 240, f"Expected 240 ratings, got {len(ratings_df)}"
    assert ratings_df.isnull().sum().sum() == 0, "Missing values detected in ratings.csv"
    print(f"✓ ratings.csv schema verified: {len(ratings_df)} ratings, 0 nulls.")

    # 3. Check users and ratings per user
    unique_users = ratings_df["user_id"].nunique()
    assert unique_users == 15, f"Expected 15 users, got {unique_users}"
    counts_per_user = ratings_df.groupby("user_id").size()
    assert (counts_per_user == 16).all(), "Not all users have exactly 16 ratings"
    print(f"✓ Exactly 15 users, each with exactly 16 ratings (all users count = 16).")

    # 4. Check rating range
    min_r = ratings_df["relevance_rating"].min()
    max_r = ratings_df["relevance_rating"].max()
    assert min_r >= 1 and max_r <= 5, f"Invalid rating range [{min_r}, {max_r}]"
    print(f"✓ Rating range verified: [{min_r}, {max_r}] (all integers 1 to 5).")

    # 5. Check duplicate user-item pairs
    dups = ratings_df.duplicated(subset=["user_id", "item_id"]).sum()
    assert dups == 0, f"Found {dups} duplicate user-item pairs!"
    print(f"✓ Duplicates check passed: 0 duplicate (user_id, item_id) pairs.")

    # 6. Check interaction matrix sparsity
    num_items = len(items_df)
    num_users = unique_users
    total_entries = num_items * num_users  # 40 * 15 = 600
    observed_entries = len(ratings_df)     # 240
    missing_entries = total_entries - observed_entries
    sparsity = missing_entries / total_entries
    print(f"✓ Total matrix cells: {num_users} users x {num_items} items = {total_entries}")
    print(f"✓ Observed ratings: {observed_entries}")
    print(f"✓ Missing entries: {missing_entries} ({sparsity * 100:.2f}% sparsity)")
    assert np.isclose(sparsity, 0.60), f"Sparsity {sparsity} is not 60%!"

    # 7. Check ratings per item
    ratings_per_item = ratings_df.groupby("item_id").size()
    print(f"✓ Ratings per scholarship: min = {ratings_per_item.min()}, max = {ratings_per_item.max()}, mean = {ratings_per_item.mean():.1f}")
    assert ratings_per_item.min() >= 2, "Some items have fewer than 2 ratings!"

    # 8. Check undergraduate CS AI summer exchange representation
    target_match = items_df[
        items_df["eligibility_text"].str.contains("undergraduate", case=False)
        & items_df["eligibility_text"].str.contains("computer science", case=False)
        & items_df["eligibility_text"].str.contains("summer", case=False)
    ]
    print(f"✓ Target query representation: found {len(target_match)} undergraduate CS summer exchange scholarships.")
    assert len(target_match) >= 3, "Insufficient target scholarships for Test Case 1"

    print("=" * 60)
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


def generate_and_save():
    """Generates and writes data/items.csv and data/ratings.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    scholarships = get_scholarships_data()
    items_df = pd.DataFrame(scholarships)

    profiles = get_user_profiles()
    ratings_df = generate_ratings(items_df, profiles)

    # Save to CSV
    items_path = os.path.join(DATA_DIR, "items.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    items_df.to_csv(items_path, index=False)
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Saved: {items_path}")
    print(f"Saved: {ratings_path}")

    # Validate
    validate_dataset(items_df, ratings_df)
    return items_df, ratings_df


if __name__ == "__main__":
    generate_and_save()
