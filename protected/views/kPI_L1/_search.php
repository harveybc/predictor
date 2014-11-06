<div class="wide form">

<?php $form=$this->beginWidget('CActiveForm', array(
        'action'=>Yii::app()->createUrl($this->route),
        'method'=>'get',
)); ?>

        <div class="row">
                <?php echo $form->label($model,'id'); ?>
                <?php echo $form->textField($model,'id',array('size'=>10,'maxlength'=>10)); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Fecha'); ?>
                <?php echo $form->textField($model,'Fecha'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Eff_Shift'); ?>
                <?php echo $form->textField($model,'Eff_Shift'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Count_Fill_Batch'); ?>
                <?php echo $form->textField($model,'Count_Fill_Batch'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Count_Pall_Batch'); ?>
                <?php echo $form->textField($model,'Count_Pall_Batch'); ?>
        </div>

        <div class="row">
                <?php echo $form->label($model,'Count_Fill_Shift'); ?>
                <?php echo $form->textField($model,'Count_Fill_Shift'); ?>
        </div>

        <div class="row buttons">
                <?php echo CHtml::submitButton(Yii::t('app', 'Search')); ?>
        </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
