# Pipeline Service - Orchestrates analysis workflows
from datetime import datetime
from typing import List
from uuid import UUID
import structlog

from app.db.session import async_session_maker
from app.models.db import Dataset, AnalysisJob, Customer, Offering, Transaction, Insight
from app.layers.l1_ingestion.parser import parse_file
from app.layers.l2_classification.role_detector import classify_dataset_role
from app.layers.l3_schema_mapping.mapper import generate_column_mappings
from app.layers.l4_offering.detector import detect_offering_types
from app.layers.l8_rules.engine import evaluate_rules

logger = structlog.get_logger()


async def run_classification_job(
    job_id: UUID,
    dataset_ids: List[UUID],
    tenant_id: UUID
):
    """Run classification for multiple datasets."""
    async with async_session_maker() as db:
        try:
            # Get job
            from sqlalchemy import select
            result = await db.execute(
                select(AnalysisJob).where(AnalysisJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                logger.error("Job not found", job_id=str(job_id))
                return
            
            # Update status
            job.status = "processing"
            job.started_at = datetime.utcnow()
            await db.commit()
            
            classifications = []
            total = len(dataset_ids)
            
            for i, dataset_id in enumerate(dataset_ids):
                # Update progress
                job.progress = int((i / total) * 100)
                job.current_step = f"Classifying dataset {i+1} of {total}"
                await db.commit()
                
                # Get dataset
                result = await db.execute(
                    select(Dataset).where(Dataset.id == dataset_id)
                )
                dataset = result.scalar_one_or_none()
                
                if not dataset:
                    continue
                
                # Classify
                classification = await classify_dataset_role(
                    dataset.file_path, 
                    dataset.file_type
                )
                
                # Update dataset
                dataset.role = classification["role"]
                dataset.role_confidence = classification["confidence"]
                dataset.date_range_start = classification.get("date_range_start")
                dataset.date_range_end = classification.get("date_range_end")
                dataset.status = "classified"
                
                classifications.append({
                    "dataset_id": str(dataset_id),
                    "detected_role": classification["role"],
                    "role_confidence": classification["confidence"],
                    "date_range": {
                        "start": str(classification.get("date_range_start")) if classification.get("date_range_start") else None,
                        "end": str(classification.get("date_range_end")) if classification.get("date_range_end") else None
                    },
                    "dominant_entities": classification.get("entities", []),
                    "requires_confirmation": classification["confidence"] < 0.8
                })
            
            # Complete job
            job.status = "completed"
            job.progress = 100
            job.current_step = "Classification complete"
            job.completed_at = datetime.utcnow()
            job.result = {"classifications": classifications}
            
            await db.commit()
            
            logger.info("Classification job completed", job_id=str(job_id))
            
        except Exception as e:
            logger.error("Classification job failed", job_id=str(job_id), error=str(e))
            
            # Update job with error
            result = await db.execute(
                select(AnalysisJob).where(AnalysisJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                await db.commit()


async def run_full_analysis_pipeline(
    job_id: UUID,
    dataset_ids: List[UUID],
    tenant_id: UUID
):
    """Run the full analysis pipeline."""
    async with async_session_maker() as db:
        try:
            from sqlalchemy import select
            
            # Get job
            result = await db.execute(
                select(AnalysisJob).where(AnalysisJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                return
            
            # Update status
            job.status = "processing"
            job.started_at = datetime.utcnow()
            await db.commit()
            
            # Step 1: Load and normalize data (20%)
            job.progress = 5
            job.current_step = "Loading datasets"
            await db.commit()
            
            datasets = {}
            for dataset_id in dataset_ids:
                result = await db.execute(
                    select(Dataset).where(Dataset.id == dataset_id)
                )
                dataset = result.scalar_one_or_none()
                if dataset:
                    df = await parse_file(dataset.file_path, dataset.file_type)
                    datasets[str(dataset_id)] = {
                        "df": df,
                        "role": dataset.role,
                        "mappings": {}  # Would be populated from ColumnMapping
                    }
            
            job.progress = 20
            job.current_step = "Normalizing data"
            await db.commit()
            
            # Step 2: Extract entities (40%)
            # In a full implementation, this would:
            # - Apply column mappings
            # - Create/update Customer records
            # - Create/update Offering records
            # - Create Transaction records
            
            job.progress = 40
            job.current_step = "Extracting entities"
            await db.commit()
            
            # Step 3: Compute analytics (60%)
            job.progress = 60
            job.current_step = "Computing analytics"
            await db.commit()
            
            # Step 4: Generate insights (80%)
            job.progress = 80
            job.current_step = "Generating insights"
            await db.commit()
            
            # In a full implementation, this would:
            # - Calculate metrics per offering
            # - Run rule engine against offerings
            # - Create Insight records
            
            # Step 5: Complete
            job.progress = 100
            job.status = "completed"
            job.current_step = "Analysis complete"
            job.completed_at = datetime.utcnow()
            job.result = {
                "datasets_processed": len(dataset_ids),
                "status": "success"
            }
            
            await db.commit()
            
            logger.info("Analysis pipeline completed", job_id=str(job_id))
            
        except Exception as e:
            logger.error("Analysis pipeline failed", job_id=str(job_id), error=str(e))
            
            result = await db.execute(
                select(AnalysisJob).where(AnalysisJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                await db.commit()
