<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

        <div class="row">
                <?php echo $form->label($model,'id'); ?>
                <?php echo $form->textField($model,'id',array('size'=>20,'maxlength'=>20)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'usuario'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'documento'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'operacion'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'autorizado'); ?>
                <?php echo $form->checkBox($model,'autorizado'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
