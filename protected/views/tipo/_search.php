<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

        <div class="row">
                <?php echo $form->label($model,'Tipo_Aceite'); ?>
                <?php echo $form->textField($model,'Tipo_Aceite',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Proceso'); ?>
                <?php echo $form->textField($model,'Proceso',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
