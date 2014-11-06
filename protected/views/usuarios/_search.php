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
                <?php echo $form->label($model,'Username'); ?>
                <?php echo $form->textField($model,'Username',array('size'=>60,'maxlength'=>128)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Analista'); ?>
                <?php echo $form->textField($model,'Analista',array('size'=>60,'maxlength'=>128)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Proceso'); ?>
                <?php echo $form->textField($model,'Proceso',array('size'=>60,'maxlength'=>128)); ?>
        </div>
  
        <div class="row">
                <?php echo $form->label($model,'Es_administrador'); ?>
                <?php echo $form->checkBox($model,'Es_administrador'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
