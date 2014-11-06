<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

        <div class="row">
                <?php echo $form->label($model,'Toma'); ?>
                <?php echo $form->textField($model,'Toma',array('size'=>20,'maxlength'=>20)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'TAG'); ?>
                <?php echo $form->textField($model,'TAG',array('size'=>50,'maxlength'=>50)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Fecha'); ?>
                <?php echo $form->textField($model,'Fecha'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'A050'); ?>
                <?php echo $form->textField($model,'A050'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'A1'); ?>
                <?php echo $form->textField($model,'A1'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'B050'); ?>
                <?php echo $form->textField($model,'B050'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'B1'); ?>
                <?php echo $form->textField($model,'B1'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'C050'); ?>
                <?php echo $form->textField($model,'C050'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'C1'); ?>
                <?php echo $form->textField($model,'C1'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'OT'); ?>
                <?php echo $form->textField($model,'OT'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
