import os
import sys

# MUST be set before importing pydeequ
os.environ["SPARK_VERSION"] = "3.3"

# Ensure driver + workers use the same python (your conda env)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pydeequ
from pyspark.sql import SparkSession
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationSuite


def test_dataset():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("data-test")
        .config("spark.jars.packages", pydeequ.DEEQU_MAVEN_COORD)  # pydeequ 1.1.0 uses DEEQU_MAVEN_COORD
        .getOrCreate()
    )

    data = [
        ("img1.jpg", "cat", 224, 224),
        ("img2.jpg", "dog", 224, 224),
        ("img3.jpg", "cat", 224, 224),
    ]
    df = spark.createDataFrame(data, ["image_path", "label", "width", "height"])

    result = (
        VerificationSuite(spark)
        .onData(df)
        .addCheck(
            Check(spark, CheckLevel.Error, "dataset validation")
            .hasSize(lambda x: x > 0)
            .isComplete("image_path")
            .isUnique("image_path")
            .isContainedIn("label", ["cat", "dog"])
            .isNonNegative("width")
            .isNonNegative("height")
        )
        .run()
    )

    spark.stop()
    assert result.status == "Success"
