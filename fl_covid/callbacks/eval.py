from ..utils.eval import evaluate
import tensorflow as tf


class Evaluate:
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        model,
        generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        tensorboard_writer=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            model            : The model for evaluation (should be prediction model)
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard_writer      : Instance of  tf.summary.FileWriter used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.model = model
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.writer     = tensorboard_writer
        self.weighted_average= weighted_average
        self.verbose         = verbose

        super(Evaluate, self).__init__()

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise ValueError('logs is none')
        logs = logs or {}

        # run evaluation
        average_precisions, TP, FP, Recall, Precision,num_annotations = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations), self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision), flush=True)
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar("mAP", self.mean_ap, step=epoch)
                tf.summary.scalar("TP", TP, step=epoch)
                tf.summary.scalar("FP", FP, step=epoch)
                tf.summary.scalar("num_annotations", num_annotations, step=epoch)
                tf.summary.scalar("FP-rate", FP/num_annotations, step=epoch)
                tf.summary.scalar("Recall", Recall, step=epoch)
                tf.summary.scalar("Precision", Precision, step=epoch)
                self.writer.flush()

        logs['mAP'] = self.mean_ap
        logs['TP'] = TP
        logs['FP'] = FP
        logs['Recall'] = Recall
        logs['Precision'] = Precision

        if self.verbose == 1:
            print('mAP: {:.4f} TP:{} FP:{} FP-rate{:.4f} Recall: {:.4f} Precision: {:.4f}'.format(self.mean_ap, TP, FP, FP/num_annotations, Recall, Precision), flush=True)

        return logs

class Evaluate_separate:
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        model,
        generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        tensorboard_writer=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            model            : The model for evaluation (should be prediction model)
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard_writer      : Instance of  tf.summary.FileWriter used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.model = model
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.writer     = tensorboard_writer
        self.weighted_average= weighted_average
        self.verbose         = verbose

        super(Evaluate_separate, self).__init__()

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise ValueError('logs is none')
        logs = logs or {}

        client_map = []
        client_tp = []
        client_fp = []
        client_recall = []
        client_precision = []

        for client in range(len(self.generator)):
        # run evaluation
            generator = self.generator[client]
            average_precisions, TP, FP, Recall, Precision, num_annotations = evaluate(
                generator,
                self.model,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                save_path=self.save_path
            )

            # compute per class average precision
            total_instances = []
            precisions = []
            for label, (average_precision, num_annotations) in average_precisions.items():
                if self.verbose == 1:
                    print('{:.0f} instances of class'.format(num_annotations), generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision), flush=True)
                total_instances.append(num_annotations)
                precisions.append(average_precision)
            if self.weighted_average:
                self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
            else:
                self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.scalar("site {} mAP".format(client) , self.mean_ap, step=epoch)
                    tf.summary.scalar("site {} TP".format(client) , TP, step=epoch)
                    tf.summary.scalar("site {} FP".format(client) , FP, step=epoch)
                    tf.summary.scalar("site {} num_annotations".format(client) , num_annotations, step=epoch)
                    tf.summary.scalar("site {} FP-rate".format(client) , FP/num_annotations, step=epoch)
                    tf.summary.scalar("site {} Recall".format(client) , Recall, step=epoch)
                    tf.summary.scalar("site {} Precision".format(client) , Precision, step=epoch)
                    self.writer.flush()

            logs['mAP'] = self.mean_ap
            logs['TP'] = TP
            logs['FP'] = FP
            logs['Recall'] = Recall
            logs['Precision'] = Precision

            if self.verbose == 1:
                print('Site {} performance:  mAP: {:.4f} TP:{} FP:{} FP-rate {:.4f}Recall: {:.4f} Precision: {:.4f}\n'.format(client, self.mean_ap, TP, FP, FP/num_annotations, Recall, Precision), flush=True)

            client_map.append(self.mean_ap)
            client_tp.append(TP)
            client_fp.append(FP)
            client_recall.append(Recall)
            client_precision.append(Precision)
            
        print('Overall performance:  mAP: {:.4f} TP:{} FP:{} Recall: {:.4f} Precision: {:.4f}\n'.format(sum(client_map)/len(client_map), sum(client_tp)/len(client_tp), sum(client_fp)/len(client_fp), sum(client_recall)/len(client_recall), sum(client_precision)/len(client_precision)), flush=True)
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar("overall mAP", sum(client_map)/len(client_map), step=epoch)
                tf.summary.scalar("overall TP", sum(client_tp)/len(client_tp), step=epoch)
                tf.summary.scalar("overall FP", sum(client_fp)/len(client_fp), step=epoch)
                tf.summary.scalar("overall Recall", sum(client_recall)/len(client_recall), step=epoch)
                tf.summary.scalar("overall Precision", sum(client_precision)/len(client_precision), step=epoch)
        return logs
