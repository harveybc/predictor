<div class="wide form">

    <?php
    $form = $this->beginWidget('CActiveForm', array(
        'action' => Yii::app()->createUrl($this->route),
        'method' => 'get',
            ));
    ?>



    <div class="row">
<?php echo $form->label($model, 'Proceso'); ?>
<?php echo $form->textField($model, 'Proceso', array('size' => 50, 'maxlength' => 50)); ?>
    </div>

    <div class="row">
<?php echo $form->label($model, 'Area'); ?>
<?php echo $form->textField($model, 'Area', array('size' => 50, 'maxlength' => 50)); ?>
    </div>

    <div class="row">
<?php echo $form->label($model, 'Equipo'); ?>
        <?php echo $form->textField($model, 'Equipo', array('size' => 50, 'maxlength' => 50)); ?>
    </div>
    <div class="row">
<?php echo $form->label($model, 'Codigo'); ?>
<?php echo $form->textField($model, 'Codigo'); ?>
    </div>





    <div class="row buttons">
    <?php echo CHtml::submitButton(Yii::t('app', 'Buscar')); ?>
    </div>

<?php $this->endWidget(); ?>

</div><!-- search-form -->
