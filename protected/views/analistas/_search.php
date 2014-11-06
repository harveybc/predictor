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
                <?php echo $form->label($model,'Analista'); ?>
                <?php echo $form->textField($model,'Analista',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Proceso'); ?>
                <?php echo $form->textField($model,'Proceso',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Pto_trabajo'); ?>
                <?php echo $form->textField($model,'Pto_trabajo',array('size'=>8,'maxlength'=>8)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'modulo'); ?>
                <?php echo $form->textField($model,'modulo'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
