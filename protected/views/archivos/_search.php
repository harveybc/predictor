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
                <?php echo $form->label($model,'nombre'); ?>
                <?php echo $form->textField($model,'nombre',array('size'=>60,'maxlength'=>512)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'tipo'); ?>
                <?php echo $form->textField($model,'tipo',array('size'=>60,'maxlength'=>64)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'tamano'); ?>
                <?php echo $form->textField($model,'tamano',array('size'=>20,'maxlength'=>20)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'contenido'); ?>
                <?php echo $form->textField($model,'contenido'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Search')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
