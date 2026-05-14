# ============================================================
# dataset_fix3.py
# Same unified multimodal dataset, with optional frozen-BERT feature caching
# ============================================================

import hashlib
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdchem
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_ENST_START = "ENST00000343813.5"
DEFAULT_ENST_END = "ENST00000349048.4"


def safe_read_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"❌ 无法解析 CSV 编码: {path}")


class ENSTPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


class MoleculePreprocessor:
    def __init__(self, max_atoms: int = 100, atom_feature_dim: int = 16):
        self.max_atoms = max_atoms
        self.atom_feature_dim = atom_feature_dim

    def _safe_3d_embed(self, mol: Chem.Mol):
        mol = Chem.Mol(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42

        ok = AllChem.EmbedMolecule(mol, params)
        if ok == 0:
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            return np.asarray(coords, dtype=np.float32), mol

        try:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, 0.0])
            return np.asarray(coords, dtype=np.float32), mol
        except Exception:
            return np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32), mol

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.size == 0:
            return coords
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        center = coords.mean(axis=0, keepdims=True)
        coords = coords - center
        max_abs = float(np.max(np.abs(coords))) if coords.size > 0 else 0.0
        if max_abs > 20.0:
            coords = coords * (20.0 / max_abs)
        return coords.astype(np.float32)

    def _atom_feature(self, atom: Chem.Atom) -> np.ndarray:
        feat = [
            atom.GetAtomicNum() / 100.0,
            atom.GetDegree() / 8.0,
            atom.GetTotalValence() / 8.0,
            atom.GetFormalCharge() / 8.0,
            atom.GetNumRadicalElectrons() / 4.0,
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs(includeNeighbors=True) / 8.0,
            float(atom.IsInRing()),
            atom.GetMass() / 250.0,
            float(atom.GetChiralTag()),
            float(atom.GetHybridization()),
            self._safe_get_implicit_valence(atom) / 8.0,
            self._safe_get_explicit_valence(atom) / 8.0,
            float(atom.HasProp("_CIPCode")) if atom.HasProp("_CIPCode") else 0.0,
            float(atom.GetIsotope()) / 300.0,
            float(atom.GetNoImplicit()),
        ]
        feat = feat[: self.atom_feature_dim]
        if len(feat) < self.atom_feature_dim:
            feat.extend([0.0] * (self.atom_feature_dim - len(feat)))
        return np.asarray(feat, dtype=np.float32)

    def smiles_to_graph(self, smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(smiles, str) or smiles.strip() == "":
            return self._empty_graph()

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._empty_graph()

            coords, mol = self._safe_3d_embed(mol)
            coords = self._normalize_coords(coords)
            atom_feats = np.asarray([self._atom_feature(atom) for atom in mol.GetAtoms()], dtype=np.float32)
            atom_feats = np.nan_to_num(atom_feats, nan=0.0, posinf=0.0, neginf=0.0)
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return self._pad_graph(atom_feats, coords)
        except Exception:
            return self._empty_graph()

    def _pad_graph(self, atom_feats: np.ndarray, coords: np.ndarray):
        n_atoms = min(len(atom_feats), self.max_atoms)
        atom_feat_pad = np.zeros((self.max_atoms, self.atom_feature_dim), dtype=np.float32)
        coord_pad = np.zeros((self.max_atoms, 3), dtype=np.float32)
        atom_mask = np.zeros((self.max_atoms,), dtype=np.float32)

        if n_atoms > 0:
            atom_feat_pad[:n_atoms] = atom_feats[:n_atoms]
            coord_pad[:n_atoms] = coords[:n_atoms]
            atom_mask[:n_atoms] = 1.0

        return atom_feat_pad, coord_pad, atom_mask

    def _empty_graph(self):
        return (
            np.zeros((self.max_atoms, self.atom_feature_dim), dtype=np.float32),
            np.zeros((self.max_atoms, 3), dtype=np.float32),
            np.zeros((self.max_atoms,), dtype=np.float32),
        )

    def _safe_get_implicit_valence(self, atom: Chem.Atom) -> float:
        try:
            return float(atom.GetValence(rdchem.ValenceType.IMPLICIT))
        except Exception:
            return 0.0

    def _safe_get_explicit_valence(self, atom: Chem.Atom) -> float:
        try:
            return float(atom.GetValence(rdchem.ValenceType.EXPLICIT))
        except Exception:
            return 0.0


class TextTokenizerCache:
    def __init__(self, model_name: str = "bert-base-uncased", max_len: int = 128):
        self.model_name = model_name
        self.max_len = max_len
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        if pd.isna(text):
            text = ""
        encoded = self.tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="np",
        )
        return encoded["input_ids"][0].astype(np.int64), encoded["attention_mask"][0].astype(np.int64)


class FrozenTextFeatureBank:
    def __init__(self, model_name: str = "bert-base-uncased", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cpu")

        config = None
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            if hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    config=config,
                    attn_implementation="eager",
                    local_files_only=True,
                )
            except TypeError:
                self.model = AutoModel.from_pretrained(model_name, config=config, local_files_only=True)
                if hasattr(self.model.config, "_attn_implementation"):
                    self.model.config._attn_implementation = "eager"
        except Exception as e:
            raise RuntimeError(f"❌ 无法从本地缓存加载冻结 BERT 以构建文本特征缓存: {e}")

        if hasattr(self.model, "set_attn_implementation"):
            try:
                self.model.set_attn_implementation("eager")
            except Exception:
                pass

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode_unique_texts(self, texts: List[str], tokenizer: TextTokenizerCache) -> Tuple[np.ndarray, np.ndarray]:
        clean_texts = ["" if pd.isna(t) else str(t) for t in texts]
        unique_texts = list(dict.fromkeys(clean_texts))
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        index_arr = np.asarray([text_to_idx[t] for t in clean_texts], dtype=np.int64)

        bank = np.zeros((len(unique_texts), tokenizer.max_len, self.hidden_size), dtype=np.float32)
        if len(unique_texts) == 0:
            return bank, index_arr

        for start in tqdm(range(0, len(unique_texts), self.batch_size), desc="Caching frozen BERT text bank"):
            chunk = unique_texts[start:start + self.batch_size]
            enc = tokenizer.tokenizer(
                chunk,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc).last_hidden_state.detach().cpu().float().numpy()
            bank[start:start + len(chunk)] = out

        return bank, index_arr


class EnhancedMolDataset(Dataset):
    SINGLE_ID_CANDIDATES = ["Drug Id", "DrugID", "Drug_Id", "drug_id"]
    SINGLE_CELL_CANDIDATES = ["Cell line name", "CellLine", "cell_line_name", "cell_line"]
    SINGLE_SMILES_CANDIDATES = ["smiles", "SMILES", "drug_smiles"]
    SINGLE_TEXT_CANDIDATES = ["description", "Description", "drug_description"]
    SINGLE_LABEL_CANDIDATES = ["IC50", "ic50", "label", "response"]

    COMBO_CELL_CANDIDATES = ["CellLine", "Cell line name", "cell_line", "cell_line_name"]
    COMBO_SMILES1_CANDIDATES = ["Drug1_SMILES", "drug1_smiles", "smiles_1"]
    COMBO_SMILES2_CANDIDATES = ["Drug2_SMILES", "drug2_smiles", "smiles_2"]
    COMBO_TEXT1_CANDIDATES = ["Drug1_Description", "drug1_description", "description_1"]
    COMBO_TEXT2_CANDIDATES = ["Drug2_Description", "drug2_description", "description_2"]
    COMBO_LABEL_CANDIDATES = ["Synergy_Score", "synergy_score", "Synergy_score", "synergy", "S_score", "label"]

    def __init__(
        self,
        csv_path: str,
        enst_start_col: str = DEFAULT_ENST_START,
        enst_end_col: str = DEFAULT_ENST_END,
        mode: str = "auto",
        preprocessed_dir: Optional[str] = None,
        reload: bool = False,
        text_model_name: str = "bert-base-uncased",
        max_atoms: int = 100,
        max_text_len: int = 128,
        label_col: Optional[str] = None,
        cache_text_embeddings: bool = False,
        text_cache_batch_size: int = 32,
    ):
        self.csv_path = csv_path
        self.df = safe_read_csv(csv_path)
        self.mode = self._infer_mode(mode)
        self.label_col = self._infer_label_col(label_col)

        if enst_start_col not in self.df.columns or enst_end_col not in self.df.columns:
            raise ValueError(f"❌ 转录本边界列不存在。请检查 {enst_start_col} 和 {enst_end_col} 是否在 CSV 中。")

        self.enst_cols = self.df.loc[:, enst_start_col:enst_end_col].columns.tolist()
        if len(self.enst_cols) != 954:
            raise ValueError(f"❌ 期望 954 个转录本特征，实际检测到 {len(self.enst_cols)} 个。")

        self.max_atoms = max_atoms
        self.max_text_len = max_text_len
        self.text_model_name = text_model_name
        self.cache_text_embeddings = bool(cache_text_embeddings)
        self.text_cache_batch_size = int(text_cache_batch_size)

        self.mol_proc = MoleculePreprocessor(max_atoms=max_atoms)
        self.text_proc = TextTokenizerCache(model_name=text_model_name, max_len=max_text_len)
        self.text_bank_builder = None
        if self.cache_text_embeddings:
            self.text_bank_builder = FrozenTextFeatureBank(model_name=text_model_name, batch_size=text_cache_batch_size)
        self.enst_proc = ENSTPreprocessor()

        self.atom_feature_dim = self.mol_proc.atom_feature_dim
        self.preprocessed_dir = preprocessed_dir
        self.cache_path = self._build_cache_path(preprocessed_dir)

        if self.cache_path is not None and os.path.exists(self.cache_path) and not reload:
            self._load()
        else:
            self._process()

    def _build_cache_path(self, preprocessed_dir: Optional[str]) -> Optional[str]:
        if preprocessed_dir is None:
            return None
        os.makedirs(preprocessed_dir, exist_ok=True)
        base = os.path.basename(self.csv_path)
        key = f"{os.path.abspath(self.csv_path)}|{self.mode}|{self.text_model_name}|{self.max_atoms}|{self.max_text_len}|cache_text={self.cache_text_embeddings}"
        key = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return os.path.join(preprocessed_dir, f"cache_{base}_{self.mode}_{key}.pt")

    def _infer_mode(self, mode: str) -> str:
        if mode != "auto":
            return mode
        cols = set(self.df.columns)
        if "Drug1_SMILES" in cols and "Drug2_SMILES" in cols:
            return "combo"
        lower_cols = {c.lower() for c in cols}
        if {"drug1_smiles", "drug2_smiles"}.issubset(lower_cols):
            return "combo"
        return "single"

    def _find_existing_col(self, candidates):
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def _infer_label_col(self, label_col: Optional[str]) -> str:
        if label_col is not None:
            if label_col not in self.df.columns:
                raise ValueError(f"❌ 指定的标签列不存在: {label_col}")
            return label_col
        col = self._find_existing_col(self.SINGLE_LABEL_CANDIDATES if self.mode == "single" else self.COMBO_LABEL_CANDIDATES)
        if col is None:
            raise ValueError("❌ 无法自动识别标签列，请通过 --label_col 显式指定。")
        return col

    def _get_single_cols(self):
        id_col = self._find_existing_col(self.SINGLE_ID_CANDIDATES)
        cell_col = self._find_existing_col(self.SINGLE_CELL_CANDIDATES)
        smiles_col = self._find_existing_col(self.SINGLE_SMILES_CANDIDATES)
        text_col = self._find_existing_col(self.SINGLE_TEXT_CANDIDATES)
        if smiles_col is None or text_col is None:
            raise ValueError("❌ 单药模式下未找到 smiles / description 列。")
        return id_col, cell_col, smiles_col, text_col

    def _get_combo_cols(self):
        cell_col = self._find_existing_col(self.COMBO_CELL_CANDIDATES)
        s1 = self._find_existing_col(self.COMBO_SMILES1_CANDIDATES)
        s2 = self._find_existing_col(self.COMBO_SMILES2_CANDIDATES)
        t1 = self._find_existing_col(self.COMBO_TEXT1_CANDIDATES)
        t2 = self._find_existing_col(self.COMBO_TEXT2_CANDIDATES)
        if None in [s1, s2, t1, t2]:
            raise ValueError("❌ 多药模式下未找到 drug1/drug2 的 smiles / description 列。")
        return cell_col, s1, s2, t1, t2

    def _safe_meta_value(self, value, fallback=""):
        if pd.isna(value):
            return fallback
        return str(value)

    def _process_drug(self, smiles: str, description: str) -> Dict[str, np.ndarray]:
        atom_feat, atom_coords, atom_mask = self.mol_proc.smiles_to_graph(smiles)
        input_ids, attention_mask = self.text_proc.encode(description)
        return {
            "atom_feat": atom_feat,
            "atom_coords": atom_coords,
            "atom_mask": atom_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _attach_frozen_text_bank_single(self, cache: Dict, descriptions: List[str]):
        if self.text_bank_builder is None:
            return
        bank, index_arr = self.text_bank_builder.encode_unique_texts(descriptions, self.text_proc)
        cache["text_hidden_bank"] = bank.astype(np.float32)
        cache["text_hidden_index"] = index_arr.astype(np.int64)
        print(f"✅ Frozen text bank cached: {len(bank)} unique descriptions")

    def _attach_frozen_text_bank_combo(self, cache: Dict, desc1: List[str], desc2: List[str]):
        if self.text_bank_builder is None:
            return
        bank1, index1 = self.text_bank_builder.encode_unique_texts(desc1, self.text_proc)
        bank2, index2 = self.text_bank_builder.encode_unique_texts(desc2, self.text_proc)
        cache["text_hidden_bank_1"] = bank1.astype(np.float32)
        cache["text_hidden_index_1"] = index1.astype(np.int64)
        cache["text_hidden_bank_2"] = bank2.astype(np.float32)
        cache["text_hidden_index_2"] = index2.astype(np.int64)
        print(f"✅ Frozen text banks cached: drug1={len(bank1)} unique, drug2={len(bank2)} unique")

    def _process(self):
        enst_raw = self.df[self.enst_cols].values.astype(np.float32)
        self.enst_proc.fit(enst_raw)
        cell_feat = self.enst_proc.transform(enst_raw)

        cache = {
            "mode": self.mode,
            "cell_feat": cell_feat,
            "labels": [],
            "meta_sample_index": [],
            "meta_sample_id": [],
            "meta_cell_line": [],
            "cache_text_embeddings": self.cache_text_embeddings,
        }

        if self.mode == "single":
            cache.update({
                "atom_feat": [], "atom_coords": [], "atom_mask": [],
                "input_ids": [], "attention_mask": [], "meta_smiles": []
            })
            id_col, cell_col, smiles_col, text_col = self._get_single_cols()
            desc_list = []
        else:
            cache.update({
                "atom_feat_1": [], "atom_coords_1": [], "atom_mask_1": [], "input_ids_1": [], "attention_mask_1": [],
                "atom_feat_2": [], "atom_coords_2": [], "atom_mask_2": [], "input_ids_2": [], "attention_mask_2": [],
                "meta_smiles_1": [], "meta_smiles_2": []
            })
            cell_col, s1, s2, t1, t2 = self._get_combo_cols()
            desc_list_1, desc_list_2 = [], []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Caching {self.mode} dataset"):
            cache["labels"].append(np.float32(row[self.label_col]))
            cache["meta_sample_index"].append(int(idx))

            if self.mode == "single":
                cache["meta_sample_id"].append(self._safe_meta_value(row[id_col], fallback=str(idx)) if id_col else str(idx))
                cache["meta_cell_line"].append(self._safe_meta_value(row[cell_col]) if cell_col else "")
                cache["meta_smiles"].append(self._safe_meta_value(row[smiles_col]))
                desc_list.append("" if pd.isna(row[text_col]) else str(row[text_col]))

                drug = self._process_drug(row[smiles_col], row[text_col])
                for k, v in drug.items():
                    cache[k].append(v)
            else:
                cache["meta_sample_id"].append(str(idx))
                cache["meta_cell_line"].append(self._safe_meta_value(row[cell_col]) if cell_col else "")
                cache["meta_smiles_1"].append(self._safe_meta_value(row[s1]))
                cache["meta_smiles_2"].append(self._safe_meta_value(row[s2]))
                desc_list_1.append("" if pd.isna(row[t1]) else str(row[t1]))
                desc_list_2.append("" if pd.isna(row[t2]) else str(row[t2]))

                drug1 = self._process_drug(row[s1], row[t1])
                drug2 = self._process_drug(row[s2], row[t2])
                for k, v in drug1.items():
                    cache[f"{k}_1"].append(v)
                for k, v in drug2.items():
                    cache[f"{k}_2"].append(v)

        cache["labels"] = np.asarray(cache["labels"], dtype=np.float32)
        cache["cell_feat"] = np.asarray(cache["cell_feat"], dtype=np.float32)
        cache["meta_sample_index"] = np.asarray(cache["meta_sample_index"], dtype=np.int64)

        for key, value in list(cache.items()):
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                cache[key] = np.stack(value, axis=0)

        if self.cache_text_embeddings:
            if self.mode == "single":
                self._attach_frozen_text_bank_single(cache, desc_list)
            else:
                self._attach_frozen_text_bank_combo(cache, desc_list_1, desc_list_2)

        self.cache = cache

        if self.cache_path is not None:
            torch.save({
                "cache": self.cache,
                "enst_mean": self.enst_proc.mean,
                "enst_std": self.enst_proc.std,
                "atom_feature_dim": self.atom_feature_dim,
            }, self.cache_path)

    def _load(self):
        payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        self.cache = payload["cache"]
        self.enst_proc.mean = payload.get("enst_mean")
        self.enst_proc.std = payload.get("enst_std")
        self.atom_feature_dim = int(payload.get("atom_feature_dim", self.atom_feature_dim))
        self.cache_text_embeddings = bool(self.cache.get("cache_text_embeddings", self.cache_text_embeddings))

    def __len__(self):
        return len(self.cache["labels"])

    def __getitem__(self, idx: int):
        item = {
            "mode": self.mode,
            "task_id": 0 if self.mode == "single" else 1,
            "label": torch.tensor(self.cache["labels"][idx], dtype=torch.float32).view(1),
            "cell_feat": torch.tensor(self.cache["cell_feat"][idx], dtype=torch.float32),
            "meta_sample_index": int(self.cache["meta_sample_index"][idx]),
            "meta_sample_id": self.cache["meta_sample_id"][idx],
            "meta_cell_line": self.cache["meta_cell_line"][idx],
        }

        if self.mode == "single":
            item.update({
                "atom_feat": torch.tensor(self.cache["atom_feat"][idx], dtype=torch.float32),
                "atom_coords": torch.tensor(self.cache["atom_coords"][idx], dtype=torch.float32),
                "atom_mask": torch.tensor(self.cache["atom_mask"][idx], dtype=torch.float32),
                "attention_mask": torch.tensor(self.cache["attention_mask"][idx], dtype=torch.long),
                "meta_smiles": self.cache["meta_smiles"][idx],
            })
            item["input_ids"] = torch.tensor(self.cache["input_ids"][idx], dtype=torch.long)
            if self.cache_text_embeddings and "text_hidden_bank" in self.cache:
                text_idx = int(self.cache["text_hidden_index"][idx])
                item["text_hidden"] = torch.tensor(self.cache["text_hidden_bank"][text_idx], dtype=torch.float32)
        else:
            item.update({
                "atom_feat_1": torch.tensor(self.cache["atom_feat_1"][idx], dtype=torch.float32),
                "atom_coords_1": torch.tensor(self.cache["atom_coords_1"][idx], dtype=torch.float32),
                "atom_mask_1": torch.tensor(self.cache["atom_mask_1"][idx], dtype=torch.float32),
                "attention_mask_1": torch.tensor(self.cache["attention_mask_1"][idx], dtype=torch.long),
                "atom_feat_2": torch.tensor(self.cache["atom_feat_2"][idx], dtype=torch.float32),
                "atom_coords_2": torch.tensor(self.cache["atom_coords_2"][idx], dtype=torch.float32),
                "atom_mask_2": torch.tensor(self.cache["atom_mask_2"][idx], dtype=torch.float32),
                "attention_mask_2": torch.tensor(self.cache["attention_mask_2"][idx], dtype=torch.long),
                "meta_smiles_1": self.cache["meta_smiles_1"][idx],
                "meta_smiles_2": self.cache["meta_smiles_2"][idx],
            })
            item["input_ids_1"] = torch.tensor(self.cache["input_ids_1"][idx], dtype=torch.long)
            item["input_ids_2"] = torch.tensor(self.cache["input_ids_2"][idx], dtype=torch.long)
            if self.cache_text_embeddings and "text_hidden_bank_1" in self.cache:
                text_idx_1 = int(self.cache["text_hidden_index_1"][idx])
                text_idx_2 = int(self.cache["text_hidden_index_2"][idx])
                item["text_hidden_1"] = torch.tensor(self.cache["text_hidden_bank_1"][text_idx_1], dtype=torch.float32)
                item["text_hidden_2"] = torch.tensor(self.cache["text_hidden_bank_2"][text_idx_2], dtype=torch.float32)
        return item



def _resolve_dataset_mode(dataset) -> str:
    if hasattr(dataset, "mode"):
        return getattr(dataset, "mode")
    if hasattr(dataset, "dataset"):
        return _resolve_dataset_mode(dataset.dataset)
    return "unknown"


class JointDataset(Dataset):
    def __init__(self, single_dataset: Dataset, combo_dataset: Dataset):
        single_mode = _resolve_dataset_mode(single_dataset)
        combo_mode = _resolve_dataset_mode(combo_dataset)
        if single_mode != "single":
            raise ValueError(f"❌ JointDataset 期望 single_dataset 为单药数据，实际模式: {single_mode}")
        if combo_mode != "combo":
            raise ValueError(f"❌ JointDataset 期望 combo_dataset 为多药数据，实际模式: {combo_mode}")

        self.single_dataset = single_dataset
        self.combo_dataset = combo_dataset
        self.mode = "joint"
        self.single_len = len(single_dataset)
        self.combo_len = len(combo_dataset)
        self.samples = [("single", i) for i in range(self.single_len)] + [("combo", i) for i in range(self.combo_len)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        task_name, real_idx = self.samples[idx]
        if task_name == "single":
            item = dict(self.single_dataset[real_idx])
        else:
            item = dict(self.combo_dataset[real_idx])
        item["joint_task_name"] = task_name
        return item


def collate_fn_joint(batch):
    single_items = [item for item in batch if int(item.get("task_id", -1)) == 0]
    combo_items = [item for item in batch if int(item.get("task_id", -1)) == 1]

    return {
        "single_batch": default_collate(single_items) if len(single_items) > 0 else None,
        "combo_batch": default_collate(combo_items) if len(combo_items) > 0 else None,
        "single_count": len(single_items),
        "combo_count": len(combo_items),
    }
