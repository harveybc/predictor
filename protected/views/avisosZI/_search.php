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
                <?php echo $form->label($model,'Ruta'); ?>
                <?php echo $form->textField($model,'Ruta',array('size'=>60,'maxlength'=>128)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Operador'); ?>
                <?php echo $form->textField($model,'Operador',array('size'=>60,'maxlength'=>64)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Fecha'); ?>
                <?php echo $form->textField($model,'Fecha'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Estado'); ?>
                <?php echo $form->textField($model,'Estado'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Observaciones'); ?>
                <?php echo $form->textField($model,'Observaciones',array('size'=>60,'maxlength'=>255)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'arreglado'); ?>
                <?php echo $form->checkBox($model,'arreglado'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'plan_mant'); ?>
                <?php echo $form->textField($model,'plan_mant'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'OT'); ?>
                <?php echo $form->textField($model,'OT'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Search')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
