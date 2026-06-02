from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class FieldStatus(str, Enum):
    FOUND = "FOUND"
    MISSING = "MISSING"
    PENDING = "PENDING"
    CONFLICT = "CONFLICT"
    FLAGGED = "FLAGGED"

class ClinicalField(BaseModel):
    value: Optional[str] = None
    status: FieldStatus = FieldStatus.MISSING
    source: Optional[str] = None       # e.g. "Page 1, admission note"
    flag_reason: Optional[str] = None  # why it was flagged

    def display(self) -> str:
        if self.status == FieldStatus.FOUND:
            return f"{self.value}  [src: {self.source}]"
        elif self.status == FieldStatus.MISSING:
            return "[MISSING — requires clinician review]"
        elif self.status == FieldStatus.PENDING:
            return f"[PENDING — {self.flag_reason or 'result not yet available'}]"
        elif self.status == FieldStatus.CONFLICT:
            return f"[CONFLICT — {self.flag_reason}]"
        elif self.status == FieldStatus.FLAGGED:
            return f"{self.value or ''}  [FLAG: {self.flag_reason}]"
        return str(self.value)

class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    status: str = "unchanged"   # added / stopped / changed / unchanged
    change_reason: Optional[str] = None
    flag: Optional[str] = None  # if change has no documented reason

class DischargeSummary(BaseModel):
    patient_id: str
    demographics: ClinicalField = Field(default_factory=ClinicalField)
    admission_date: ClinicalField = Field(default_factory=ClinicalField)
    discharge_date: ClinicalField = Field(default_factory=ClinicalField)
    principal_diagnosis: ClinicalField = Field(default_factory=ClinicalField)
    secondary_diagnoses: ClinicalField = Field(default_factory=ClinicalField)
    hospital_course: ClinicalField = Field(default_factory=ClinicalField)
    procedures: ClinicalField = Field(default_factory=ClinicalField)
    discharge_medications: List[Medication] = Field(default_factory=list)
    allergies: ClinicalField = Field(default_factory=ClinicalField)
    follow_up: ClinicalField = Field(default_factory=ClinicalField)
    pending_results: ClinicalField = Field(default_factory=ClinicalField)
    discharge_condition: ClinicalField = Field(default_factory=ClinicalField)
    clinician_flags: List[str] = Field(default_factory=list)

    def to_readable(self) -> str:
        lines = [
            "=" * 70,
            f"DISCHARGE SUMMARY DRAFT — {self.patient_id}",
            "[!]  THIS IS A DRAFT FOR CLINICIAN REVIEW — NOT FINAL",
            "=" * 70,
            f"DEMOGRAPHICS       : {self.demographics.display()}",
            f"ADMISSION DATE     : {self.admission_date.display()}",
            f"DISCHARGE DATE     : {self.discharge_date.display()}",
            f"PRINCIPAL DIAGNOSIS: {self.principal_diagnosis.display()}",
            f"SECONDARY DX       : {self.secondary_diagnoses.display()}",
            "",
            "HOSPITAL COURSE:",
            self.hospital_course.display(),
            "",
            f"PROCEDURES         : {self.procedures.display()}",
            "",
            "DISCHARGE MEDICATIONS:",
        ]
        if self.discharge_medications:
            for m in self.discharge_medications:
                flag_str = f"  [!] {m.flag}" if m.flag else ""
                lines.append(
                    f"  • {m.name} {m.dose or ''} {m.route or ''} {m.frequency or ''} "
                    f"[{m.status.upper()}]{flag_str}"
                )
        else:
            lines.append("  [MISSING — requires clinician review]")
        lines += [
            "",
            f"ALLERGIES          : {self.allergies.display()}",
            f"FOLLOW-UP          : {self.follow_up.display()}",
            f"PENDING RESULTS    : {self.pending_results.display()}",
            f"DISCHARGE CONDITION: {self.discharge_condition.display()}",
        ]
        if self.clinician_flags:
            lines += ["", "CLINICIAN FLAGS (require review):"]
            for flag in self.clinician_flags:
                lines.append(f"  [!]  {flag}")
        lines.append("=" * 70)
        return "\n".join(lines)
