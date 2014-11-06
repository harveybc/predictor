<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
		<?php echo $form->labelEx($model,'Ruta'); ?>
<?php echo $form->textField($model,'Ruta',array('size'=>60,'maxlength'=>128)); ?>
<?php echo $form->error($model,'Ruta'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Operador'); ?>
<?php echo $form->textField($model,'Operador',array('size'=>60,'maxlength'=>64)); ?>
<?php echo $form->error($model,'Operador'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Fecha'); ?>
<?php echo $form->textField($model,'Fecha'); ?>
<?php echo $form->error($model,'Fecha'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Estado'); ?>
<?php echo $form->textField($model,'Estado'); ?>
<?php echo $form->error($model,'Estado'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Observaciones'); ?>
<?php echo $form->textField($model,'Observaciones',array('size'=>60,'maxlength'=>255)); ?>
<?php echo $form->error($model,'Observaciones'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'arreglado'); ?>
<?php echo $form->checkBox($model,'arreglado'); ?>
<?php echo $form->error($model,'arreglado'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'plan_mant'); ?>
<?php echo $form->textField($model,'plan_mant'); ?>
<?php echo $form->error($model,'plan_mant'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'Codigo'); ?>
<?php echo $form->textField($model,'Codigo'); ?>
<?php echo $form->error($model,'Codigo'); ?>
	</div>
	<div class="row">
		<?php echo $form->labelEx($model,'OT'); ?>
<?php echo $form->textField($model,'OT'); ?>
<?php echo $form->error($model,'OT'); ?>
	</div>


