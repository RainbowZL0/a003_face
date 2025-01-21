from a000_CONFIG import (
    DATASET_SF_TL54_PATH,
    TRAINING_NUM_SAMPLES_PER_EPOCH,
    DATASET_SF_TL54_CROPPED_PATH,
    TRAINING_DETECTOR_NAME,
)
from a002_model.a003_training.a002_DatasetForTraining import DatasetForTrainingAndVali
from a002_model.a003_training.a003_MyTrainingObj import (
    MyTrainingObj,
)


def start_main_train():
    dataset_for_training_and_vali = DatasetForTrainingAndVali(
        original_dataset_path=DATASET_SF_TL54_PATH,
        num_samples_per_epoch=TRAINING_NUM_SAMPLES_PER_EPOCH,
        create_or_exist_cropped_dataset_at_path=DATASET_SF_TL54_CROPPED_PATH,
        training_detector_name=TRAINING_DETECTOR_NAME,
        probability_for_mod_choices_for_training_dict=None,
        whether_build_cropped_dataset=False,
    )
    my_training_obj = MyTrainingObj(dataset_for_training_and_vali)

    my_training_obj.start_train_and_vali()

    # my_training_obj.load_my_state()
    # my_training_obj.high_level_api_for_vali_and_analyze()

    # analyze_detailed_result_to_get_cosine_similarity_distribution(
    #     detailed_result_json_path=os.path.join(VALI_LOG_FOLDER, VALI_ANALYZE_USING_DETAILED_RESULT_JS_NAME)
    # )


if __name__ == '__main__':
    start_main_train()
    pass
