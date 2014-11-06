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
                <?php echo $form->textField($model,'usuario',array('size'=>31,'maxlength'=>31)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'modulo'); ?>
                <?php echo $form->textField($model,'modulo',array('size'=>32,'maxlength'=>32)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'operacion'); ?>
                <?php echo $form->textField($model,'operacion',array('size'=>32,'maxlength'=>32)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'ip'); ?>
                <?php echo $form->textField($model,'ip',array('size'=>12,'maxlength'=>12)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'descripcion'); ?>
                <?php echo $form->textField($model,'descripcion',array('size'=>60,'maxlength'=>255)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'fecha'); ?>
                <?php echo $form->textField($model,'fecha'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Search')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
