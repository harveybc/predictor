<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
			</div>

	<div class="row">
			</div>


<label for="Usuarios">Belonging Usuarios</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'usuario0',
							'fields' => 'Username',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
							)
						); ?>
			<label for="Competencias">Belonging Competencias</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'competencia0',
							'fields' => 'descripcion',
							'allowEmpty' => true,
							'style' => 'dropdownlist',
							)
						); ?>
			