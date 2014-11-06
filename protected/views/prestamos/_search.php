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
                <?php echo $form->label($model,'metaDoc'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'cedula'); ?>
                <?php echo $form->textField($model,'cedula',array('size'=>32,'maxlength'=>32)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'usuario'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'usuarioRcv'); ?>
                <?php ; ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fechaPrestamo'); ?>
                <?php echo $form->textField($model,'fechaPrestamo'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fechaDevolucion'); ?>
                <?php echo $form->textField($model,'fechaDevolucion'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'observaciones'); ?>
                <?php echo $form->textField($model,'observaciones',array('size'=>60,'maxlength'=>128)); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
